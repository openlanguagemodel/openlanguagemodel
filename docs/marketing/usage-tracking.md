# Usage Tracking

Use this page to track whether launch, SEO, university outreach, and docs work
are turning into actual usage.

## Weekly Metrics

Record these once per week:

- GitHub stars, forks, watchers, open issues, and pull requests
- GitHub traffic: clones, views, and referrers from **Insights -> Traffic**
- PyPI downloads for the last day, week, and month
- Google Search Console impressions, clicks, average position, and top queries
- Website pages with the most visits
- Posts, emails, classes, or talks that produced visible traffic

## PyPI Downloads

PyPI Stats provides package download totals and daily time series through a JSON
API. Its data excludes known mirrors, is updated daily, and time series data is
retained for 180 days.

Useful endpoints:

- `https://pypistats.org/api/packages/openlanguagemodel/recent`
- `https://pypistats.org/api/packages/openlanguagemodel/overall?mirrors=false`

You can also capture a lightweight JSON snapshot from the repo:

```bash
GH_TOKEN="$(gh auth token)" python scripts/usage_snapshot.py
```

## GitHub Traffic

GitHub's traffic page is useful for short-term launch analysis. Check:

- Referring sites after launch posts
- Popular content paths after documentation or course outreach
- Clone count after PyPI, Colab, or class announcements

## Search Console

Set up Google Search Console for:

- `https://openlanguagemodel.github.io/openlanguagemodel/`

Submit:

- `https://openlanguagemodel.github.io/openlanguagemodel/sitemap.xml`

Track these queries first:

- `openlanguagemodel`
- `openlanguagemodel github`
- `PyTorch LLM library`
- `train language model from scratch`
- `custom transformer architecture PyTorch`
- `FineWeb-Edu training PyTorch`

## Launch Attribution

For each public post or university outreach batch, record:

- date
- channel
- link
- audience
- GitHub star delta after 24 hours and 7 days
- PyPI download delta after 24 hours and 7 days
- top referrers and Search Console query changes
