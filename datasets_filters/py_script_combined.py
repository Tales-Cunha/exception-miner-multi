import os
import pandas as pd

# Lista de repositórios (use a coluna 'repo' do CSV acima)
repos = [
"flask","youtube-dl","models","keras","ansible","scikit-learn","fastapi","manim",
"requests","transformers","scrapy","public-apis","system-design-primer","python",
"awesome-python","python-100-days","thefuck","hellogithub","django",
"you-get"
]

repo_stats_dir = "/home/talescunha/Jairo/exception-miner-multi/results/parser/py"
out_combined = "/home/talescunha/Jairo/exception-miner-multi/tmp/py_stats_combined_2.csv"   # change if you want

dfs = []
for repo in repos:
    # map repo name to expected filename (some repos use dashes/underscores — adapt if necessary)
    fname = os.path.join(repo_stats_dir, f"{repo}_stats.csv")
    if not os.path.exists(fname):
        # try alternative with repo name replaced (e.g., dashes in repo)
        alt = os.path.join(repo_stats_dir, f"{repo.replace('-', '_')}_stats.csv")
        if os.path.exists(alt):
            fname = alt
        else:
            # also try lower-case / other heuristics if you expect different names
            print("Not found:", fname)
            continue
    try:
        df = pd.read_csv(fname)
        df['source_repo'] = repo
        dfs.append(df)
        print("Loaded:", fname, "rows:", len(df))
    except Exception as e:
        print("Error reading", fname, e)

if not dfs:
    print("No files loaded. Check paths.")
else:
    combined = pd.concat(dfs, ignore_index=True)

    # Desired balanced sample size per class
    target_each = 351

    if 'n_try_except' not in combined.columns:
        print("Column 'n_try_except' not found in combined dataframe. Saving full combined file instead.")
        combined.to_csv(out_combined, index=False)
        print("Combined CSV saved to", out_combined, "total rows:", len(combined))
    else:
        pos = combined[combined['n_try_except'] == 1]
        neg = combined[combined['n_try_except'] == 0]

        def sample_or_resample(df, n, rng=42):
            if len(df) == 0:
                return pd.DataFrame(columns=df.columns)
            # If enough rows, sample without replacement; otherwise sample with replacement
            replace = len(df) < n
            return df.sample(n=n, replace=replace, random_state=rng)

        pos_sample = sample_or_resample(pos, target_each, rng=42)
        neg_sample = sample_or_resample(neg, target_each, rng=43)

        final = pd.concat([pos_sample, neg_sample], ignore_index=True)
        final = final.sample(frac=1, random_state=44).reset_index(drop=True)

        final.to_csv(out_combined, index=False)
        print("Balanced CSV saved to", out_combined, "total rows:", len(final))
        print("n_try_except counts -> 1:", len(final[final['n_try_except'] == 1]), ", 0:", len(final[final['n_try_except'] == 0]))