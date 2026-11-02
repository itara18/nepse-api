from app.loader import load_csv

CSV = (
    "Date,Open,High,Low,Close,Volume\n"
    '2025-01-01,100,105,99,102,"1,000"\n'
    "2025-01-02,102,106,101,104,1200\n"
)


def test_load_csv_inserts_rows(db_session, tmp_path):
    path = tmp_path / "NABIL.csv"
    path.write_text(CSV)
    assert load_csv(db_session, "NABIL", path) == 2


def test_load_csv_is_idempotent(db_session, tmp_path):
    path = tmp_path / "NABIL.csv"
    path.write_text(CSV)
    load_csv(db_session, "NABIL", path)
    assert load_csv(db_session, "NABIL", path) == 0


def test_read_price_csv_cleans_and_sorts(tmp_path):
    from app.loader import read_price_csv

    path = tmp_path / "NABIL.csv"
    path.write_text(
        "Date,Open,High,Low,Close,Volume\n"
        "2025-01-02,102,106,101,104,1200\n"
        '2025-01-01,100,105,99,102,"1,000"\n'
    )
    df = read_price_csv(path)
    assert list(df["volume"]) == [1000, 1200]
    assert str(df["date"].iloc[0]) == "2025-01-01"
