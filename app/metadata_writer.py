from __future__ import annotations

import io
import embedded_xmp
from stage_progress import track
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable
import xml.etree.ElementTree as ET

from runtime_safety import atomic_path, file_snapshot, _UNCHECKED

_XML_LOCK = threading.RLock()


XMPMETA_NS = "adobe:ns:meta/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
DC_NS = "http://purl.org/dc/elements/1.1/"
LR_NS = "http://ns.adobe.com/lightroom/1.0/"
XMP_NS = "http://ns.adobe.com/xap/1.0/"
IPTC_NS = "http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
CFF_NS = "https://childfacefinder.local/ns/1.0/"

# Stable prefixes make generated sidecars readable and compatible with Adobe tools.
ET.register_namespace("x", XMPMETA_NS)
ET.register_namespace("rdf", RDF_NS)
ET.register_namespace("dc", DC_NS)
ET.register_namespace("lr", LR_NS)
ET.register_namespace("xmp", XMP_NS)
ET.register_namespace("Iptc4xmpExt", IPTC_NS)
ET.register_namespace("cff", CFF_NS)


@dataclass(slots=True)
class XmpExportSummary:
    requested_files: int = 0
    written_files: int = 0
    unchanged_files: int = 0
    failed_files: int = 0
    failed_photos: set[Path] = field(default_factory=set)
    errors: dict[Path, str] = field(default_factory=dict)


class XmpMetadataWriter:
    """Merge XMP into supported image containers, otherwise Adobe sidecars."""

    def __init__(self, log: Callable[[str], None] | None = None) -> None:
        self.log = log or (lambda _text: None)

    @staticmethod
    def _sidecar_path(image_path: Path) -> Path:
        # Standard RAW/Adobe sidecar naming: IMG_0001.CR3 -> IMG_0001.xmp.
        wanted = image_path.with_suffix(".xmp")
        if wanted.parent.exists():
            matches = [p for p in wanted.parent.iterdir() if p.name.casefold() == wanted.name.casefold()]
            if len(matches) > 1:
                raise ValueError(f"Несколько XMP с разным регистром имени: {wanted}")
            if matches:
                return matches[0]
        return wanted

    @staticmethod
    def _remember_namespaces(data: bytes) -> None:
        """Preserve existing prefixes as far as ElementTree allows."""
        try:
            for _event, item in ET.iterparse(io.BytesIO(data), events=("start-ns",)):
                prefix, uri = item
                # ElementTree reserves ns\d+ prefixes internally.
                if prefix and prefix.startswith("ns") and prefix[2:].isdigit():
                    continue
                try:
                    ET.register_namespace(prefix or "", uri)
                except ValueError:
                    pass
        except ET.ParseError:
            # The real parse below will report the useful error.
            pass

    @staticmethod
    def _new_tree() -> ET.ElementTree:
        root = ET.Element(f"{{{XMPMETA_NS}}}xmpmeta")
        rdf = ET.SubElement(root, f"{{{RDF_NS}}}RDF")
        ET.SubElement(rdf, f"{{{RDF_NS}}}Description")
        return ET.ElementTree(root)

    @classmethod
    def _load_tree(cls, path: Path, data=_UNCHECKED) -> ET.ElementTree:
        if data is _UNCHECKED:
            data = file_snapshot(path)
        if data is None:
            return cls._new_tree()
        cls._remember_namespaces(data)
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True, insert_pis=True))
        root = ET.fromstring(data, parser=parser)
        return ET.ElementTree(root)

    @staticmethod
    def _description(root: ET.Element, tag: str | None = None) -> ET.Element:
        rdf = root if root.tag == f"{{{RDF_NS}}}RDF" else root.find(f"{{{RDF_NS}}}RDF")
        if rdf is None:
            raise ValueError("XMP не содержит корректного rdf:RDF; исходник сохранён")
        descriptions = [d for d in rdf.findall(f"{{{RDF_NS}}}Description")
                        if d.get(f"{{{RDF_NS}}}about", "") == ""]
        if tag:
            matches = [d for d in descriptions if d.find(tag) is not None or tag in d.attrib]
            if len(matches) > 1 or any(len(d.findall(tag)) > 1 for d in matches):
                raise ValueError("XMP содержит несколько конфликтующих полей ключевых слов")
            if matches:
                return matches[0]
        if descriptions:
            return descriptions[0]
        return ET.SubElement(rdf, f"{{{RDF_NS}}}Description", {f"{{{RDF_NS}}}about": ""})

    @staticmethod
    def _bag_values(description: ET.Element, tag: str) -> tuple[ET.Element, list[str]]:
        container = description.find(tag)
        if container is None:
            container = ET.SubElement(description, tag)
        scalar = description.attrib.pop(tag, None)
        bag = container.find(f"{{{RDF_NS}}}Bag")
        if bag is None:
            children = [node for node in container if isinstance(node.tag, str)]
            if len(children) == 1 and children[0].tag == f"{{{RDF_NS}}}Seq":
                bag = children[0]
                bag.tag = f"{{{RDF_NS}}}Bag"
            elif children or container.attrib:
                raise ValueError("Неизвестная структура ключевых слов XMP; исходник сохранён")
            else:
                scalar = scalar or (container.text or '').strip()
                container.text = None
                bag = ET.SubElement(container, f"{{{RDF_NS}}}Bag")
        elif len([c for c in container if isinstance(c.tag, str)]) != 1:
            raise ValueError("Неоднозначная структура ключевых слов XMP")
        if scalar and not any(li.text == scalar for li in bag):
            ET.SubElement(bag, f"{{{RDF_NS}}}li").text = scalar
        values = [(li.text or "").strip() for li in bag.findall(f"{{{RDF_NS}}}li") if (li.text or "").strip()]
        return bag, values

    @staticmethod
    def _append_unique(bag: ET.Element, existing: list[str], additions: Iterable[str]) -> bool:
        changed = False
        seen = {value.casefold() for value in existing}
        for raw in additions:
            value = " ".join(str(raw).split())
            if any(not (ord(c) in (9, 10, 13) or 0x20 <= ord(c) <= 0xD7FF or 0xE000 <= ord(c) <= 0xFFFD or 0x10000 <= ord(c) <= 0x10FFFF) for c in value):
                raise ValueError("Недопустимые символы в имени для XMP")
            if not value or value.casefold() in seen:
                continue
            li = ET.SubElement(bag, f"{{{RDF_NS}}}li")
            li.text = value
            seen.add(value.casefold())
            changed = True
        return changed

    @staticmethod
    def _remove_value(bag: ET.Element, value: str) -> bool:
        changed = False
        wanted = value.casefold()
        for li in list(bag.findall(f"{{{RDF_NS}}}li")):
            if (li.text or "").strip().casefold() == wanted:
                bag.remove(li)
                changed = True
        return changed

    @classmethod
    def _merge(cls, tree: ET.ElementTree, people: Iterable[str], is_best: bool) -> bool:
        root = tree.getroot()
        description = cls._description(root)
        people_sorted = sorted({" ".join(p.split()) for p in people if p.strip()}, key=str.casefold)
        changed = False

        # Person tags are deliberately add-only: a user's existing keyword must
        # never be deleted just because a later recognition run differs.
        if people_sorted:
            subject_bag, subject_values = cls._bag_values(cls._description(root, f"{{{DC_NS}}}subject"), f"{{{DC_NS}}}subject")
            changed |= cls._append_unique(subject_bag, subject_values, people_sorted)

            hierarchical_bag, hierarchical_values = cls._bag_values(
                cls._description(root, f"{{{LR_NS}}}hierarchicalSubject"), f"{{{LR_NS}}}hierarchicalSubject"
            )
            hierarchy = [f"People|{person}" for person in people_sorted]
            changed |= cls._append_unique(hierarchical_bag, hierarchical_values, hierarchy)

        # Best is a ChildFaceFinder-owned marker, so unlike human names it can
        # safely be synchronized. This prevents stale Best flags after a rerun.
        description = cls._description(root, f"{{{LR_NS}}}hierarchicalSubject")
        hierarchical = description.find(f"{{{LR_NS}}}hierarchicalSubject")
        bag = None if hierarchical is None else hierarchical.find(f"{{{RDF_NS}}}Bag")
        marker = "ChildFaceFinder|Best"
        has_marker = False
        if bag is not None:
            has_marker = any(
                (li.text or "").strip().casefold() == marker.casefold()
                for li in bag.findall(f"{{{RDF_NS}}}li")
            )
        if is_best and not has_marker:
            if bag is None:
                bag, _values = cls._bag_values(description, f"{{{LR_NS}}}hierarchicalSubject")
            changed |= cls._append_unique(bag, [], [marker])
        elif not is_best and has_marker and bag is not None:
            changed |= cls._remove_value(bag, marker)

        return changed

    @classmethod
    def _sync_people(cls, tree, people):
        root = tree.getroot()
        before = ET.tostring(root)
        subject_tag, hierarchy_tag = f"{{{DC_NS}}}subject", f"{{{LR_NS}}}hierarchicalSubject"
        person_tag, owned_tag = f"{{{IPTC_NS}}}PersonInImage", f"{{{CFF_NS}}}People"
        old = set()
        for tag in (person_tag, owned_tag):
            desc = cls._description(root, tag)
            if desc.find(tag) is not None or tag in desc.attrib:
                bag, values = cls._bag_values(desc, tag)
                old.update(values)
                for li in list(bag):
                    bag.remove(li)
        desc = cls._description(root, hierarchy_tag)
        if desc.find(hierarchy_tag) is not None or hierarchy_tag in desc.attrib:
            bag, values = cls._bag_values(desc, hierarchy_tag)
            for value in values:
                if value.startswith('People|'):
                    old.add(value[len('People|'):])
                    old.add(value.rsplit('|', 1)[-1])
                    cls._remove_value(bag, value)
            # Retire the old script-owned keyword; selection now uses xmp:Label.
            cls._remove_value(bag, 'ChildFaceFinder|Best')
        desc = cls._description(root, subject_tag)
        if desc.find(subject_tag) is not None or subject_tag in desc.attrib:
            bag, _ = cls._bag_values(desc, subject_tag)
            for name in old:
                cls._remove_value(bag, name)
        people = sorted({' '.join(p.split()) for p in people if p.strip()}, key=str.casefold)
        for tag, values in ((subject_tag, people), (hierarchy_tag, ['People|' + p for p in people]),
                            (person_tag, people), (owned_tag, people)):
            if values:
                bag, current = cls._bag_values(cls._description(root, tag), tag)
                cls._append_unique(bag, current, values)
        return ET.tostring(root) != before

    @classmethod
    def _sync_label(cls, tree, label):
        root = tree.getroot()
        before = ET.tostring(root)
        tag = f"{{{XMP_NS}}}Label"
        rdf = root if root.tag == f"{{{RDF_NS}}}RDF" else root.find(f"{{{RDF_NS}}}RDF")
        if rdf is None:
            raise ValueError('Некорректный XMP: отсутствует RDF')
        # Change only the label of this image, never nested region descriptions.
        for desc in rdf.findall(f"{{{RDF_NS}}}Description"):
            if desc.get(f"{{{RDF_NS}}}about", ''):
                continue
            desc.attrib.pop(tag, None)
            for node in desc.findall(tag):
                desc.remove(node)
        if label:
            if any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in label):
                raise ValueError('Некорректное название цветовой метки')
            ET.SubElement(cls._description(root), tag).text = label
        return ET.tostring(root) != before

    @staticmethod
    def _write_atomic(tree: ET.ElementTree, path: Path, *, expected=_UNCHECKED, check_cancel=None) -> None:
        with atomic_path(path, expected=expected, cancel=check_cancel) as temp:
            tree.write(temp, encoding="utf-8", xml_declaration=True, short_empty_elements=True)

    def export(self, *args, **kwargs):
        # ElementTree's namespace registry is process-global.
        with _XML_LOCK:
            return self._export(*args, **kwargs)

    def _export(
        self,
        people_by_photo: dict[Path, set[str]],
        best_photos: set[Path] | None = None,
        all_photos: Iterable[Path] | None = None,
        rename_people: dict[str, str] | None = None,
        check_cancel: Callable[[], None] | None = None,
        preserve_photos: set[Path] | None = None,
        sync_people: bool = False,
        write_people: bool = True,
        sync_labels: bool = False,
        best_label: str = 'Select',
        legacy_best: bool = True,
        progress=lambda done, total, name: None,
    ) -> XmpExportSummary:
        best_photos = best_photos or set()
        candidates = set(people_by_photo) | set(all_photos or ()) | set(best_photos)
        groups = {}
        embedded = set()
        summary = XmpExportSummary()
        for image_path in track(sorted(candidates, key=lambda p: (str(p).casefold(), str(p))), progress, "Подготовка XMP", lambda p: p.name):
            if check_cancel:
                check_cancel()
            try:
                if embedded_xmp.supports_embedded(image_path):
                    if image_path.stat().st_size < embedded_xmp.MAX_WRITE_SIZE:
                        embedded.add(image_path)
                        xmp_path = image_path
                    else:
                        self.log(f"XMP: {image_path.name}: превышен лимит встроенной записи 1 ГБ; используется sidecar")
                        xmp_path = self._sidecar_path(image_path)
                else:
                    xmp_path = self._sidecar_path(image_path)
                # Group even untagged siblings so an existing shared sidecar
                # cannot receive one image's result and erase another's Best.
                groups.setdefault(str(xmp_path.resolve()).casefold(), []).append((image_path, xmp_path))
            except Exception as exc:
                summary.requested_files += 1
                summary.failed_files += 1
                summary.failed_photos.add(image_path)
                summary.errors[image_path] = str(exc)
                self.log(f"XMP: {image_path}: {exc}")
        for siblings in track(list(groups.values()), progress, "Запись XMP", lambda group: group[0][0].name):
            if check_cancel:
                check_cancel()
            image_path, xmp_path = siblings[0]
            if not any(p in people_by_photo or p in best_photos for p, _ in siblings) and not xmp_path.exists():
                continue
            summary.requested_files += 1
            if any(p in (preserve_photos or set()) for p, _ in siblings):
                summary.unchanged_files += 1
                continue
            if len(siblings) > 1:
                summary.failed_files += 1
                summary.failed_photos.update(p for p, _ in siblings)
                summary.errors.update({p: "Несколько изображений используют один sidecar" for p, _ in siblings})
                self.log("XMP пропущен: несколько изображений используют один sidecar: " + ", ".join(str(p) for p, _ in siblings))
                continue
            people = people_by_photo.get(image_path, set())
            try:
                digest = embedded_xmp.fingerprint(image_path, check_cancel) if image_path in embedded else None
                before = embedded_xmp.read(image_path) if image_path in embedded else file_snapshot(xmp_path)
                tree = self._load_tree(xmp_path, before)
                renamed = False
                # Only explicit anonymous-catalog aliases are migrated. Named
                # reference runs never supply aliases.
                for tag, prefix in ((f"{{{DC_NS}}}subject", ""),
                                    (f"{{{LR_NS}}}hierarchicalSubject", "People|")):
                    aliases = {prefix + old: prefix + new for old, new in (rename_people or {}).items()}
                    for bag in tree.getroot().findall(f".//{tag}/{{{RDF_NS}}}Bag"):
                        for li in list(bag):
                            if li.text in aliases:
                                new = aliases[li.text]
                                if any(other is not li and other.text == new for other in bag):
                                    bag.remove(li)
                                else:
                                    li.text = new
                                renamed = True
                changed = renamed
                if sync_people and write_people:
                    changed = self._sync_people(tree, people) or changed
                elif write_people:
                    changed = self._merge(tree, people, image_path in best_photos if legacy_best else False) or changed
                elif legacy_best:
                    changed = self._merge(tree, [], image_path in best_photos) or changed
                if sync_labels:
                    changed = self._sync_label(tree, best_label if image_path in best_photos else '') or changed
                if not changed:
                    summary.unchanged_files += 1
                    continue
                if image_path in embedded:
                    packet = ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True)
                    embedded_xmp.write(image_path, packet, digest, check_cancel)
                else:
                    self._write_atomic(tree, xmp_path, expected=before, check_cancel=check_cancel)
                summary.written_files += 1
            except Exception as exc:
                if check_cancel:
                    check_cancel()
                summary.failed_files += 1
                summary.failed_photos.add(image_path)
                summary.errors[image_path] = str(exc)
                self.log(f"XMP: не удалось записать {xmp_path}: {exc}")

        return summary


def build_people_by_photo(
    match_details: dict[tuple[str, str], tuple[float, str, object, Path]],
) -> dict[Path, set[str]]:
    """Invert final accepted matches into image -> people without touching matching logic."""
    result: dict[Path, set[str]] = defaultdict(set)
    for (person_id, _relative_path), (_dist, _source, _face, image_path) in match_details.items():
        result[image_path].add(person_id)
    return dict(result)
