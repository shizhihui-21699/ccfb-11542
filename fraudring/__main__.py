"""命令行入口:
  python -m fraudring generate [--accounts N] [--days D] [--rings R] [--out DIR]
  python -m fraudring ingest   [--db DB] [--dir DATA_DIR]
  python -m fraudring analyze  [--db DB] [--data-dir DATA_DIR] [--resolution R]
  python -m fraudring serve    [--db DB] [--port P]
  python -m fraudring demo     一键: 生成 -> 导入 -> 分析 (小规模)
"""
import argparse
import time

from .db import connect, ingest_dir
from .generate import generate
from .pipeline import analyze


def main():
    p = argparse.ArgumentParser(prog="fraudring")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--out", default="data/incoming")
    g.add_argument("--accounts", type=int, default=1_000_000)
    g.add_argument("--days", type=int, default=7)
    g.add_argument("--rings", type=int, default=12)
    g.add_argument("--seed", type=int, default=42)

    i = sub.add_parser("ingest")
    i.add_argument("--db", default="fraud.db")
    i.add_argument("--dir", default="data/incoming")

    a = sub.add_parser("analyze")
    a.add_argument("--db", default="fraud.db")
    a.add_argument("--data-dir", default="data/incoming")
    a.add_argument("--resolution", type=float, default=1.0)

    s = sub.add_parser("serve")
    s.add_argument("--db", default="fraud.db")
    s.add_argument("--port", type=int, default=8000)

    d = sub.add_parser("demo")
    d.add_argument("--accounts", type=int, default=200_000)
    d.add_argument("--days", type=int, default=7)
    d.add_argument("--rings", type=int, default=12)
    d.add_argument("--db", default="fraud.db")
    d.add_argument("--dir", default="data/incoming")

    args = p.parse_args()

    if args.cmd == "generate":
        generate(args.out, args.accounts, args.days, args.rings, args.seed)
    elif args.cmd == "ingest":
        db = connect(args.db)
        t0 = time.time()
        nf, nr = ingest_dir(db, args.dir)
        print(f"[ingest] 新文件={nf} 新行={nr} 耗时={time.time()-t0:.1f}s")
    elif args.cmd == "analyze":
        db = connect(args.db)
        analyze(db, data_dir=args.data_dir, resolution=args.resolution)
    elif args.cmd == "serve":
        from .webapp import create_app
        create_app(args.db).run(host="127.0.0.1", port=args.port, debug=False)
    elif args.cmd == "demo":
        generate(args.dir, args.accounts, args.days, args.rings)
        db = connect(args.db)
        ingest_dir(db, args.dir)
        analyze(db, data_dir=args.dir)
        print(f"\n查看结果: python -m fraudring serve --db {args.db} "
              f"然后打开 http://127.0.0.1:8000")


if __name__ == "__main__":
    main()
