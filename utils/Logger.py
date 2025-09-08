import sys
from pathlib import Path

from loguru import logger


class Logger:
    @staticmethod
    def init(log_name: str | Path = None) -> None:
        logger.remove()
        logger.add(sys.stdout, format='<g>{time:YYYY-MM-DD HH:mm:ss.SSS}</g> <r>|</r> <level>{level: <8}</level> <r>|</r> {message}')

        if log_name is not None:
            logger.add(
                log_name,
                format='<g>{time:YYYY-MM-DD HH:mm:ss.SSS}</g> <r>|</r> <level>{level: <8}</level> <r>|</r> {message}'
            )

    @staticmethod
    def critical(message: str) -> None:
        logger.critical(message)

    @staticmethod
    def debug(message: str) -> None:
        logger.debug(message)

    @staticmethod
    def error(message: str) -> None:
        logger.error(message)

    @staticmethod
    def info(message: str) -> None:
        logger.info(message)

    @staticmethod
    def tabulate(data: list[list[str | int | float | None]], headers: list[str] = None) -> None:
        if headers:
            data = [headers] + data

        data = [
            [round(x, 4) if isinstance(x, float) else x for x in row]
            for row in data
        ]

        col_widths = [max(len(str(row[i])) for row in data) for i in range(len(data[0]))]

        sep = '+' + '+'.join('-' * (w + 2) for w in col_widths) + '+'

        Logger.info(sep)
        for idx, row in enumerate(data):
            line = '|' + '|'.join(f' {str(cell):<{col_widths[i]}} ' for i, cell in enumerate(row)) + '|'
            Logger.info(line)
            if idx == 0 and headers:
                Logger.info(sep)
        Logger.info(sep)
