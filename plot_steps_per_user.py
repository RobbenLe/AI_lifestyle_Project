#!/usr/bin/env python3
import argparse, os
import pandas as pd
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 

def engine():
    host=os.getenv("PGHOST","localhost"); port=os.getenv("PGPORT","5432")
    db=os.getenv("PGDATABASE","postgres"); user=os.getenv("PGUSER","postgres")
    pwd=os.getenv("PGPASSWORD","")
    return create_engine(f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}")

def read_steps(user_id: int|None, since: str|None) -> pd.DataFrame:
    where=[]; params={}
    if user_id is not None: where.append("sdv_user_id = :uid"); params["uid"]=user_id
    if since: where.append("calendar_date >= :since"); params["since"]=since
    w = ("WHERE "+ " AND ".join(where)) if where else ""
    sql=f"""
      SELECT sdv_user_id::int AS user_id,
             calendar_date::date AS day,
             COALESCE(steps,0)::int AS steps,
             steps_goal
      FROM public.steps
      {w}
      ORDER BY user_id, day;
    """
    with engine().connect() as con:
        return pd.read_sql(text(sql), con, params=params)

def annotate_peaks(ax, d):
    mx=d.loc[d.steps.idxmax()]; mn=d.loc[d.steps.idxmin()]
    for row,label in [(mx,"Peak"),(mn,"Low")]:
        ax.annotate(f"{label}\n{row.steps:,}", xy=(row.day,row.steps),
                    xytext=(0,10), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

def plot_one_user(grp: pd.DataFrame, outdir: str):
    uid = int(grp["user_id"].iloc[0])

    # --- Clean & order ---
    d = grp.copy()

    # 1) Force proper datetime and drop invalid ones (NaT -> rows with bad/missing dates)
    d["day"] = pd.to_datetime(d["day"], errors="coerce")
    d = d.dropna(subset=["day"])

    # 2) Normalize to midnight (not strictly required, but keeps it tidy)
    d["day"] = d["day"].dt.tz_localize(None).dt.normalize()

    # 3) Ensure steps are numeric
    d["steps"] = pd.to_numeric(d["steps"], errors="coerce").fillna(0).astype(int)


    # 4) Sort by date
    d = d.sort_values("day").reset_index(drop=True)

    # Optional: ignore zeros if they represent missing-device days
    # d = d[d["steps"] > 0]

    # --- Moving average for trend ---
    d["ma7"] = d["steps"].rolling(7, min_periods=1).mean()

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(16, 6))

    # Bar width in "days" (0.9 ~ almost full width per-day)
    ax.bar(d["day"], d["steps"], width=0.9)

    # 7-day trend line on same axes (no explicit colors)
    ax.plot(d["day"], d["ma7"], color="tab:orange", linewidth=2, zorder=3)

    # Title/labels
    ax.set_title(f"Daily Steps (user_id={uid})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Steps")

    # --- Nice date axis: monthly ticks, YYYY-MM labels ---
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    # Optional: limit x to data span with a small margin
    ax.set_xlim(d["day"].min() - pd.Timedelta(days=1),
                d["day"].max() + pd.Timedelta(days=1))

    # --- Annotate peak/low ---
    max_idx = d["steps"].idxmax()
    min_idx = d["steps"].idxmin()
    for idx, label in [(max_idx, "Peak"), (min_idx, "Low")]:
        x = d.loc[idx, "day"]; y = d.loc[idx, "steps"]
        ax.annotate(f"{label}\n{y:,}", xy=(x, y),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    fig.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"steps_user_{uid}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def summarize(d):
    mx=d.loc[d.steps.idxmax()]; mn=d.loc[d.steps.idxmin()]
    return dict(user_id=int(d.user_id.iloc[0]), days=len(d),
                total_steps=int(d.steps.sum()), avg_steps=float(d.steps.mean()),
                peak_steps=int(mx.steps), peak_day=str(mx.day),
                lowest_steps=int(mn.steps), lowest_day=str(mn.day))

def main():
    ap=argparse.ArgumentParser("Plot steps per user from PostgreSQL")
    ap.add_argument("--user", type=int, help="only this sdv_user_id")
    ap.add_argument("--since", type=str, help="min date YYYY-MM-DD")
    ap.add_argument("--outdir", default="charts")
    args=ap.parse_args()

    df=read_steps(args.user, args.since)
    if df.empty: print("No rows match filters."); return

    rows=[]
    for uid,grp in df.groupby("user_id", sort=True):
        out=plot_one_user(grp, args.outdir)
        stats=summarize(grp); stats["chart"]=out; rows.append(stats)
        print(f"user_id={uid}: {stats}")

    pd.DataFrame(rows).to_csv(os.path.join(args.outdir,"summary.csv"), index=False)
    print(f"Saved charts to ./{args.outdir} and summary.csv")

if __name__=="__main__": main()
