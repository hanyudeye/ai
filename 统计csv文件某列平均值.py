import csv
import io
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path


PRICE_COLUMN = "价格"
OUTPUT_FILE = "价格平均值.txt"


def parse_price(value: str, file_path: Path, row_number: int) -> Decimal:
    normalized = (
        value.strip()
        .replace(",", "")
        .replace("，", "")
        .replace("¥", "")
        .replace("￥", "")
        .replace("元", "")
        .strip()
    )
    try:
        price = Decimal(normalized)
    except InvalidOperation as error:
        raise ValueError(
            f"{file_path.name} 第 {row_number} 行的价格无效：{value!r}"
        ) from error

    if not price.is_finite():
        raise ValueError(
            f"{file_path.name} 第 {row_number} 行的价格不是有限数值：{value!r}"
        )
    return price


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    csv_files = sorted(script_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit("当前目录没有 CSV 文件。")

    total = Decimal("0")
    count = 0

    for file_path in csv_files:
        try:
            content = file_path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="gb18030")

        reader = csv.DictReader(io.StringIO(content))
        if not reader.fieldnames:
            raise ValueError(f"{file_path.name} 没有表头。")

        price_column = next(
            (
                column
                for column in reader.fieldnames
                if column is not None and column.strip() == PRICE_COLUMN
            ),
            None,
        )
        if price_column is None:
            raise ValueError(f"{file_path.name} 中找不到“{PRICE_COLUMN}”列。")

        for row_number, row in enumerate(reader, start=2):
            value = (row.get(price_column) or "").strip()
            if not value:
                continue
            total += parse_price(value, file_path, row_number)
            count += 1

    if count == 0:
        raise ValueError("所有 CSV 文件的“价格”列都没有有效数值。")

    average = (total / count).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    result = (
        f"统计文件数：{len(csv_files)}\n"
        f"价格记录数：{count}\n"
        f"价格平均值：{average} 元\n"
    )
    output_path = script_dir / OUTPUT_FILE
    output_path.write_text(result, encoding="utf-8")
    print(f"统计完成，结果已写入：{output_path}")


if __name__ == "__main__":
    main()