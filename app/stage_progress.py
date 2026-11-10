"""Report actual completed work, including skipped items, without guessing time."""
def track(items, report, stage, label=lambda item: ''):
    total = len(items)
    report(0, max(1, total), stage)
    for index, item in enumerate(items):
        name = label(item)
        text = f'{stage}: {name}' if name else stage
        report(index, total, text)
        yield item
        report(index + 1, total, text)
    if not total:
        report(1, 1, stage)
