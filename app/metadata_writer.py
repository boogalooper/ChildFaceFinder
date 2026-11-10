from __future__ import annotations

import io
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
import xml.etree.ElementTree as ET


XMPMETA_NS = "adobe:ns:meta/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
DC_NS = "http://purl.org/dc/elements/1.1/"
LR_NS = "http://ns.adobe.com/lightroom/1.0/"

# Stable prefixes make generated sidecars readable and compatible with Adobe tools.
ET.register_namespace("x", XMPMETA_NS)
ET.register_namespace("rdf", RDF_NS)
ET.register_namespace("dc", DC_NS)
ET.register_namespace("lr", LR_NS)


@dataclass(slots=True)
class XmpExportSummary:
    requested_files: int = 0
    written_files: int = 0
    unchanged_files: int = 0
    failed_files: int = 0


class XmpMetadataWriter:
    """Write ChildFaceFinder results to non-destructive XMP sidecars.

    The image itself is never modified. Existing XMP sidecars are parsed and
    merged rather than replaced, so unrelated Lightroom/Camera Raw metadata is
    kept. Confident identities, including quality-rejected faces, should be passed here.
    """

    def __init__(self, log: Callable[[str], None] | None = None) -> None:
        self.log = log or (lambda _text: None)

    @staticmethod
    def _sidecar_path(image_path: Path) -> Path:
        # Standard RAW/Adobe sidecar naming: IMG_0001.CR3 -> IMG_0001.xmp.
        # This intentionally also applies to JPEG/TIFF so originals are never
        # modified by ChildFaceFinder.
        return image_path.with_suffix(".xmp")

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
    def _load_tree(cls, path: Path) -> ET.ElementTree:
        if not path.exists():
            return cls._new_tree()
        data = path.read_bytes()
        cls._remember_namespaces(data)
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True, insert_pis=True))
        root = ET.fromstring(data, parser=parser)
        return ET.ElementTree(root)

    @staticmethod
    def _description(root: ET.Element) -> ET.Element:
        description = root.find(f".//{{{RDF_NS}}}Description")
        if description is not None:
            return description
        rdf = root.find(f".//{{{RDF_NS}}}RDF")
        if rdf is None:
            rdf = ET.SubElement(root, f"{{{RDF_NS}}}RDF")
        return ET.SubElement(rdf, f"{{{RDF_NS}}}Description")

    @staticmethod
    def _bag_values(description: ET.Element, tag: str) -> tuple[ET.Element, list[str]]:
        container = description.find(tag)
        if container is None:
            container = ET.SubElement(description, tag)
        bag = container.find(f"{{{RDF_NS}}}Bag")
        if bag is None:
            # Preserve a non-Bag value if present and migrate its rdf:li values.
            existing_values = [
                (node.text or "").strip()
                for node in container.findall(f".//{{{RDF_NS}}}li")
                if (node.text or "").strip()
            ]
            for child in list(container):
                container.remove(child)
            bag = ET.SubElement(container, f"{{{RDF_NS}}}Bag")
            for value in existing_values:
                li = ET.SubElement(bag, f"{{{RDF_NS}}}li")
                li.text = value
        values = [
            (li.text or "").strip()
            for li in bag.findall(f"{{{RDF_NS}}}li")
            if (li.text or "").strip()
        ]
        return bag, values

    @staticmethod
    def _append_unique(bag: ET.Element, existing: list[str], additions: Iterable[str]) -> bool:
        changed = False
        seen = {value.casefold() for value in existing}
        for raw in additions:
            value = " ".join(str(raw).split())
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
            subject_bag, subject_values = cls._bag_values(description, f"{{{DC_NS}}}subject")
            changed |= cls._append_unique(subject_bag, subject_values, people_sorted)

            hierarchical_bag, hierarchical_values = cls._bag_values(
                description, f"{{{LR_NS}}}hierarchicalSubject"
            )
            hierarchy = [f"People|{person}" for person in people_sorted]
            changed |= cls._append_unique(hierarchical_bag, hierarchical_values, hierarchy)

        # Best is a ChildFaceFinder-owned marker, so unlike human names it can
        # safely be synchronized. This prevents stale Best flags after a rerun.
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

    @staticmethod
    def _write_atomic(tree: ET.ElementTree, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".cff.tmp")
        try:
            # ElementTree writes valid UTF-8 XML while preserving all parsed
            # unknown elements/attributes. XMP packet PIs, if present, are kept
            # by the parser above.
            tree.write(temp, encoding="utf-8", xml_declaration=True, short_empty_elements=True)
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)

    def export(
        self,
        people_by_photo: dict[Path, set[str]],
        best_photos: set[Path] | None = None,
        all_photos: Iterable[Path] | None = None,
        rename_people: dict[str, str] | None = None,
    ) -> XmpExportSummary:
        best_photos = best_photos or set()

        # Accepted matches always participate. Other photos are inspected only
        # if their sidecar already exists, solely to clear a stale CFF Best tag.
        targets = set(people_by_photo)
        if all_photos is not None:
            for image_path in all_photos:
                if self._sidecar_path(image_path).exists():
                    targets.add(image_path)

        summary = XmpExportSummary(requested_files=len(targets))
        for image_path in sorted(targets, key=lambda p: str(p).casefold()):
            xmp_path = self._sidecar_path(image_path)
            people = people_by_photo.get(image_path, set())
            try:
                tree = self._load_tree(xmp_path)
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
                changed = self._merge(tree, people, image_path in best_photos) or renamed
                if not changed:
                    summary.unchanged_files += 1
                    continue
                self._write_atomic(tree, xmp_path)
                summary.written_files += 1
            except Exception as exc:
                summary.failed_files += 1
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
