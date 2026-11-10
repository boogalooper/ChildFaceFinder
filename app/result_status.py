def result_title(summary):
    errors = summary.unreadable_photos + summary.xmp_failed + summary.file_errors
    saved = bool(summary.output_csv or summary.file_copies or summary.xmp_written or summary.xmp_unchanged)
    if errors:
        return 'Выполнено частично — есть ошибки' if saved else 'Результат не сохранён — ошибки обработки'
    if summary.destination is not None and not summary.completed_photos:
        return 'Обработка завершена — нет фотографий для раскладки'
    return 'Готово'
