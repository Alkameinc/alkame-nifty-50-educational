# ALKAME-NIFTY50
## Vision, Methodology & Business Charter

**Document status:** Living source of truth — v1.0
**Prepared for:** Founding team, future engineering collaborators, and future institutional/investor readers
**Scope:** India market entry product of Alkame Inc
**Classification:** Internal charter + external-facing methodology reference. Not a legal filing. Not investment advice.

---

## 0. Document Control

This document exists to answer one question consistently, for every audience that ever reads it: **what is Alkame-Nifty50, what does it actually do, what won't it do, and why should anyone trust it?**

It is written to serve three readers at once:

1. **The founding/engineering team** — this is the charter we build against. If a feature request conflicts with this document, the document wins until we deliberately revise it.
2. **Future investors or partners** — this is the pitch, stated honestly rather than promotionally.
3. **Future users (traders, and eventually retail investors)** — this is the methodology page that explains, in plain language, how signals are produced and why they should or shouldn't be trusted at any given moment.

Where these three audiences would want different emphasis, we've kept the underlying facts identical and only varied the framing — because a company that tells investors one story and users another isn't one we want to run.

---

## 1. Executive Summary

Alkame-Nifty50 is Alkame Inc's India-market product: an AI-driven market intelligence engine that watches the NIFTY 50 (and, on the roadmap, the broader Indian market) across dozens of distinct event categories — technical, corporate, macroeconomic, and global — and turns them into clearly-reasoned, confidence-scored, risk-first trading signals. A human always makes the final call. The system never places a trade on its own.

Alkame Inc has spent more than a decade (since 2013) building financial data infrastructure and market-analytics technology, largely behind the scenes for institutional use. Alkame-Nifty50 is the vehicle through which that decade of infrastructure work is pointed at the Indian retail and semi-professional trading market — building on the same principles the company already applies in its home market: filter noise aggressively, show the downside before the upside, and never let the system touch anyone's money directly.

This document lays out the scientific methodology, the honest regulatory picture under Indian securities law, the business model, and the expansion path from a 50-stock internal tool to a full-market platform.

---

## 2. Our Mission and the Problem We're Solving

Most people who want to invest seriously in the Indian market have a full-time job that isn't investing. They cannot watch five hundred stocks, read every exchange filing, track every RBI announcement, or notice the moment crude oil jumps and quietly starts squeezing paint and tyre manufacturers' margins. Institutions have entire desks for this. Individuals have a phone, a browser tab, and limited time.

At the same time, India's retail investing boom has a dark underside. SEBI's own research found that individual traders' net losses widened sharply in FY25, and the derivatives segment in particular is dominated by retail losses — the overwhelming majority of individual F&O traders lose money. Much of this happens because retail traders act on unregistered "tip" channels — Telegram groups, WhatsApp forwards, YouTube influencers — with no accountability, no risk disclosure, and frequently no real methodology behind the call at all.

**Our mission is to close that gap honestly**: give a disciplined, transparent, risk-first version of the market intelligence that institutions already have, to people who have a job instead of a trading desk — built by a team with the compliance posture of a real financial technology company, not an anonymous tip channel.

---

## 3. Company Lineage — From Alkame Inc to Alkame-Nifty50

| Era | What happened |
|---|---|
| 2013 | Alkame Inc founded (Delaware, USA). |
| 2013 – 2024 | Built data infrastructure, market-analytics tooling, and trading automation behind the scenes for institutional financial-market use. |
| 2025 | Alkame turned outward — bringing that infrastructure directly to individual investors for the first time, and entered the Indian market. |
| 2026 – present | **Alkame-Nifty50** is built as Alkame's India-market flagship: a ground-up, India-specific system reflecting NSE market structure, Indian trading hours, rupee-denominated risk, and Indian regulatory obligations — not a translated copy of the US product. |

Alkame's founding mission has always been to **level the field** — putting institution-grade market intelligence into the hands of people who don't have an institution behind them. Alkame-Nifty50 is that mission applied to the Indian market specifically, on Indian data, under Indian law.

---

## 4. Product Philosophy: Why This Is Not a Get-Rich-Quick Tool

This is the single most important section of this document, and it governs every product decision downstream.

**Alkame-Nifty50 is not built to make anyone a billionaire.** It is built to do something much less exciting and much more valuable: make sure the investments you already have are standing on solid ground, and give you an early, well-reasoned warning the moment something changes that ground.

Concretely, this means:

- **A signal exists to answer one question: "has anything changed that I should know about?"** — not "here's your next 10x."
- **A HOLD with a clear, honest reason is a successful output**, not a failure to produce a BUY. Silence in the face of genuine uncertainty is more valuable than manufactured conviction.
- **Every signal shows the downside before the upside.** Historical drawdown range, worst-case scenarios, and what has *not* worked in similar setups come first — not as a footnote.
- **Confidence scores are earned, not assumed.** A number is only shown to a user once it has been checked against how well past confidence scores of that type actually predicted outcomes (see Section 5). An uncalibrated confidence score is worse than no confidence score, because it creates false certainty.
- **The system is designed to protect capital first, and identify opportunity second.** If we ever have to choose between the two in a product decision, capital protection wins.

If a future version of this product ever drifts toward "exciting picks" over "honest risk assessment," that is a violation of this charter, not a pivot.

---

## 5. The Scientific Method Behind Alkame-Nifty50

Every claim in this section is a real, enforced engineering constraint in the codebase, not aspirational language. This is what "scientific" means in practice for us:

**Time-based validation, never randomized.** Models are trained and tested with `shuffle=False`, splitting strictly by time. A model that "predicts" the past using knowledge of the future is not a model, it's a leak — and randomized shuffling of time-series data is the most common way trading systems accidentally do exactly that.

**No lookahead bias, enforced at the feature level.** Every rolling-window feature (moving averages, RSI, volatility bands) is computed using `.shift(1)` — meaning a feature is only allowed to see data that would genuinely have been available at the moment of the decision. This sounds like a small implementation detail; it is in fact the single most common way retail trading systems produce backtests that look great and live results that don't.

**No confidence score without a calibration check upstream.** Before any confidence percentage is shown to a user, the system checks: historically, when we said "70% confidence," did that class of prediction actually come true roughly 70% of the time? If confidence scores are systematically overconfident or underconfident, they are corrected or suppressed — never shown raw.

**No signal without an edge check versus the NIFTY baseline.** A signal-generation approach isn't allowed to go live simply because it has positive average returns — it must show a real edge over passively holding the NIFTY 50 index itself, net of costs. If a strategy can't beat "just buy the index," it isn't adding value and shouldn't be presented as if it does.

**No backtest without slippage and transaction costs.** Every historical simulation includes realistic slippage and brokerage/STT-equivalent transaction costs. A backtest that ignores real-world friction is measuring a strategy that doesn't exist.

**Every event is scope-checked before it's applied.** A macro shock, a sector-specific move, and a single-company announcement are fundamentally different things, and treating them the same is how good systems produce bad advice. Every incoming signal — whether from price action, corporate filings, or news — is explicitly tagged as affecting the whole market, a specific sector, or a single stock, and only applied to the tickers it actually affects (see Section 6).

**Human-in-the-loop by design, not as a fallback.** See Section 7 — this isn't a limitation we intend to remove; it's a permanent design principle for the signal-generation side of the product.

---

## 6. How the Engine Works — Event Taxonomy at a Glance

Alkame-Nifty50 continuously tracks event categories grouped by how reliably they can be measured and how broadly they apply:

| Category | Examples | Scope handling |
|---|---|---|
| **Price-action & technical** | Opening range breakout, VWAP cross, RSI/MACD signals, volume spikes, volatility expansion | Always single-stock |
| **Relative/market-context** | Outperformance vs. NIFTY, sector relative strength, correlation breakdown | Stock, benchmarked against sector/index |
| **Corporate events** | Board meetings, announcements, dividends/splits/bonuses, insider & bulk/block deals | Always single-stock, sourced directly from exchange data |
| **News & sentiment** | Company headlines, sentiment-scored, India-market tagged | Defaults to single-stock; escalates to sector/market scope only when the headline itself signals a broader driver (e.g. crude oil, currency, monsoon) |
| **Global/cross-asset** | Dollar Index, Gold, Silver, Crude, USD/INR, US markets, VIX | Market-wide by default, sector-differentiated where the economics demand it (e.g. a weak rupee helps exporters and hurts importers — never treated as one blanket effect) |
| **Macro/calendar** | RBI policy decisions, GDP releases, Union Budget, elections, festive demand windows, monsoon status, FDI flows | Human-curated calendar; scoped to the sectors genuinely exposed (e.g. rate decisions tagged to banking/NBFC/auto, not to IT) |
| **Geopolitical/shock events** | War, blockade, sanctions, sudden global shocks | No structured data feed exists for this anywhere in the world — handled via an always-on background risk monitor plus an explicit human toggle (see below) |

**The Global Risk Monitor runs continuously**, computing a composite risk reading from VIX, Dollar Index, Gold, and Crude every cycle — but it never changes a single prediction by itself. When conditions look unusual, it surfaces a clear banner explaining what's driving the reading and asks the trader whether to enable risk-adjusted predictions. Nothing is auto-applied. Once enabled, the confidence downgrade this produces is proportional to how exposed each stock's sector actually is to the specific driver — an oil shock hits energy and auto stocks harder than it hits IT, and the system reflects that instead of applying one flat penalty to everything.

---

## 7. Human-in-the-Loop by Design

Mirroring the principle Alkame has applied since entering the individual-investor market: **the system tells you what it sees. You decide. You place every trade yourself, through your own broker.**

This is not a temporary limitation waiting for a future automation release — it is a foundational trust decision, for three reasons:

1. **It's the responsible design for a signal-generation product**, full stop — a system confident enough to trade your money unsupervised is a system that has stopped being honest about uncertainty.
2. **It keeps the product on the simpler, faster side of Indian securities regulation** for as long as we choose to keep it that way (see Section 9) — the moment a system executes orders automatically, an entirely different, heavier regulatory regime applies.
3. **It matches how good decisions actually get made** — a well-reasoned signal a human can accept, reject, or override, with their own market knowledge layered on top, consistently outperforms a black box no one is allowed to question.

An automated-execution mode is a legitimate future roadmap item (Section 12), never a v1 assumption.

---

## 8. What Alkame-Nifty50 Is Not

Said plainly, so there's never ambiguity internally or externally:

- **Not a promise of returns.** Investing involves risk, including loss of principal. Nothing produced by this system is a guarantee.
- **Not a broker.** We do not hold funds, execute trades, or have custody of anyone's money or securities, now or in any near-term roadmap phase.
- **Not personalized financial advice** — not yet. Until an Investment Adviser arm exists (Section 12), everything produced is general market research and signal generation for a subscriber base, not advice tailored to any individual's personal financial situation.
- **Not a black box that asks for blind trust.** Every signal comes with the reasoning behind it — which event(s) triggered it, how confident the system actually is and why, and what the honest downside looks like.
- **Not a replacement for a SEBI-registered professional** where one is legally required — see Section 9 for exactly where that line sits today and how it moves as the product grows.

---

## 9. Regulatory & Compliance Roadmap (India)

**This section is not legal advice.** It reflects our current understanding of publicly available SEBI circulars and regulations as of mid-2026, and it must be reviewed by SEBI-registered legal counsel before any public launch, fee-based offering, or expansion beyond the current internal team. We are treating this proactively rather than as an afterthought, because unregistered "tip" operations are a real, actively enforced problem in India today, and we have no intention of being one.

There are **three separate regulatory tracks** that apply at different points in the roadmap — they are often confused with each other, so we keep them explicitly distinct:

### Track 1 — SEBI Research Analyst (RA) Registration
**Governs:** publishing or distributing buy/sell/hold recommendations or target prices to any audience beyond internal, personal use.

- **Trigger point for us:** the moment Alkame-Nifty50's signals are made available to anyone outside the founding trading team — whether or not a fee is charged. SEBI evaluates substance (does this influence someone's investment decision?), not just how a disclaimer is worded.
- **Requirements (subject to confirmation with counsel at filing time):** a relevant postgraduate qualification (finance/economics/business, or equivalent professional certification such as CFA/CA) plus relevant experience; passing the NISM-Series-XV Research Analyst Certification exam (a revised version took effect January 2026); appointing a principal analyst; maintaining compliance and record-keeping systems; disclosing all conflicts of interest on every report; retaining records for a minimum of five years.
- **Where it sits today:** while Alkame-Nifty50 remains an internal decision-support tool for our own small trading team, this registration is very likely not yet triggered. It becomes necessary the moment we open the product to any outside user, paid or free — this is the single most important compliance gate on our entire roadmap, and it must be resolved *before*, not after, any public beta.
- **F&O-specific note for later phases:** the moment recommendations include options/futures, SEBI additionally expects every recommendation to carry the specific instrument, strike, expiry, entry premium range, stop-loss, and target — bare direction calls without this are treated as non-compliant regardless of registration status.

### Track 2 — SEBI Algorithmic Trading Framework
**Governs:** any system that places, modifies, or cancels exchange orders automatically, without a human confirming each order.

- **Trigger point for us:** only once we build the auto-execution feature described in Section 12 — not before. As long as a human clicks "buy" or "sell" in their own broker's app based on our signal, this framework does not currently apply to us.
- **What changes once it does:** as of April 1, 2026, this framework is fully in force. We would need to partner with a SEBI-registered broker (who becomes the "Principal," with us as the "Agent"/algo provider), every order would need to carry an exchange-assigned Algo-ID, and — because our signal logic is proprietary rather than a fully transparent execution rule — we would be classified as a "black-box" algo provider, which layers the Research Analyst requirement from Track 1 on top of the exchange empanelment process.
- **Where it sits today:** not yet applicable. Flagged here so the engineering team never accidentally builds direct broker-API order placement without knowing this gate exists.

### Track 3 — SEBI Investment Adviser (IA) Registration
**Governs:** personalized investment advice tailored to an individual client's specific financial situation, goals, and risk profile — as opposed to general research published to a subscriber base.

- **Trigger point for us:** the future wealth-management arm (Section 12) — the moment we move from "here's a signal for anyone watching this stock" to "here's what you personally should do given your goals."
- **What it requires:** separate, more stringent registration than the RA track, with fiduciary obligations and specific fee-structure rules. Notably, SEBI does not permit the same entity to freely mix advisory and commission-based distribution roles — this needs a clean structural decision well before we build it, not after.
- **Where it sits today:** not yet applicable. A clearly future phase.

**Summary posture:** today, as an internal tool used only by our own trading team, we are almost certainly outside all three frameworks. The instant we plan to let anyone else use this — even for free — Track 1 (Research Analyst registration) becomes the gating item for the entire company, and no public launch should happen ahead of resolving it with actual SEBI-registered counsel.

---

## 10. Business Model & Monetization

**All figures below are illustrative and directional — proposed starting points for customer discovery, not final pricing.** Actual pricing must be validated with real prospective users and finalized alongside Track 1 compliance (Section 9), since regulatory status directly affects what we're legally allowed to charge for and how.

### Proposed tiers

| Tier | Illustrative price | What it includes |
|---|---|---|
| **Watchlist (Free)** | ₹0 | Up to 5 self-selected NIFTY50 stocks, end-of-day digest, delayed signals — top-of-funnel, not a real-time trading tool |
| **Pro (Individual)** | ~₹1,499/month | Full NIFTY50 coverage, real-time dashboard, all event categories, calibrated confidence scores, personal signal history |
| **Trading Desk (Team)** | ~₹5,999/month per seat, or a desk bundle | Multi-user shared dashboard, shared human-insight annotations and override log, priority data refresh cadence |
| **Institutional / Partner** | Custom | White-label signal feed for small funds, RIAs, or family offices, compliance-ready audit-trail exports — contingent on Track 1 registration being complete |

### Revenue streams (in order of near-term viability)

1. **Subscription revenue** from the Pro and Trading Desk tiers — the core, near-term business.
2. **Institutional/partner licensing** of the signal feed once RA registration is in place.
3. **Future AUM-based advisory fees** once the Investment Adviser arm (Track 3) launches, structured within whatever fee limits SEBI's IA regulations prescribe at that time.
4. **Explicitly ruled out, at least for now:** advertising, selling user data, or any pay-for-visibility arrangement with listed companies — all of which would directly undermine the "risk-first, no hype" trust position this entire charter is built on.

### Market opportunity (India, current)

India's retail participation has grown rapidly: NSE crossed roughly **12.7 crore (127 million) unique registered investors by January 2026**, with total demat accounts well above 21 crore across depositories. Yet SEBI's own 2025 investor survey found that public awareness of equity investing (around 63%) far outpaces actual participation (roughly 9–9.5%) — meaning the addressable audience of aware-but-underserved individuals is enormous relative to those currently investing directly. At the same time, there are only around 1,700 SEBI-registered Research Analysts in the entire country, against an enormous and growing volume of (mostly unregistered and often illegal) stock-tip activity on Telegram, WhatsApp, and YouTube. That gap between real demand for trustworthy signals and the tiny supply of properly regulated providers is exactly the opening Alkame-Nifty50 is built to fill, on the right side of the law.

At the same time, this opportunity comes with an honest caution built into our own product philosophy: SEBI's research shows the overwhelming majority of individual traders in the derivatives segment lose money, with aggregate net losses widening sharply in FY25. That statistic is precisely why this charter insists on risk-first framing rather than opportunity-first marketing — we intend to be part of fixing that pattern, not part of what's driving it.

---

## 11. Expansion Roadmap

| Phase | Scope | Key dependency |
|---|---|---|
| **1 — Current** | NIFTY 50 cash equities, internal trading team only | Engineering build (Section 16) |
| **2** | NIFTY 500, organized by sector-specific modules rather than one flat list | Broader data pipeline scaling, refined sector mapping |
| **3** | F&O/options overlay | Track 1 registration + F&O-specific disclosure requirements (strike/expiry/premium/stop-loss/target on every recommendation) |
| **4** | Other indices — Bank Nifty, Sensex, Nifty Next 50, sectoral and midcap/smallcap indices | Phase 2 infrastructure, index-specific event tuning |
| **5** | Commodities (MCX — gold, silver, crude domestic contracts) | Extending the existing global cross-asset pipeline we already built for Gold/Silver/Crude signals into tradeable domestic contracts |
| **6** | Personalized wealth management | Track 3 (Investment Adviser) registration — structurally separate business line |
| **7** | Auto-execution | Track 2 (Algo framework) — broker partnership as Principal, Algo-ID empanelment, black-box RA overlay |
| **Longer-term** | Potential re-expansion into other markets, leveraging India build learnings back into Alkame's global product | Contingent on all of the above being proven first |

We treat this as a strict sequence, not a wish list — each phase's dependency is a real gate, not a suggestion, and Phase 3 onward should not be started until the Section 9 registration questions for that phase are actually resolved.

---

## 12. Team & Operating Model

Consistent with Alkame Inc's existing culture: a small, senior, remote-first team of engineers, data scientists, and market specialists, rather than a large undifferentiated headcount. The India arm operates out of Noida, with the same core trust commitments the parent company holds itself to everywhere it operates:

- We do not hold client funds.
- Every trade is placed by the person themselves, through their own broker.
- We are upfront about what the AI can and cannot do — including in this document.

As Alkame-Nifty50 grows past the internal-tool phase, we expect to need: a compliance officer / SEBI-registration-track owner (critical path item, per Section 9), additional ML/data engineering capacity, and — once distribution begins — a small support and business-development function, mirroring the roles Alkame's global careers page already lists for its US operation.

---

## 13. Risk Disclosure

Investing and trading involve risk, including the possible loss of principal. Past performance — including any backtested performance shown by this system — is not indicative of future results. Alkame-Nifty50 is, at this stage, an internal decision-support tool; it does not currently execute trades, hold funds, or provide personalized investment advice. All confidence scores, signals, and illustrative figures represent a model's output, not a guarantee, and are subject to the calibration and edge-check discipline described in Section 5 — which exists precisely because models can and do fail silently without it. Before this product is offered to anyone outside the founding team, it must carry accurate, SEBI-compliant disclosures appropriate to its registration status at that time, and users should independently verify the registration status of any signal or research provider — including this one, once public — via SEBI's official intermediary database.

---

## 14. Success Metrics — What We Hold Ourselves To

We will consider this product successful when:

- Signals are consistently well-calibrated (a "70% confidence" signal is right roughly 70% of the time, tracked continuously, not asserted once).
- Backtests show a real, cost-adjusted edge over the NIFTY 50 baseline before any strategy is allowed to run live.
- Users make **better-informed decisions**, including deciding *not* to trade — a quiet, correct HOLD is counted as a win internally, not a miss.
- No signal is ever shown without a clear, honest account of its downside.
- We remain compliant with whichever SEBI track applies to our current phase, with no gap between what we're doing and what we're registered to do.

We will consider it a failure — regardless of engagement metrics or revenue — if the product ever overstates certainty, hides its downside, or operates a public offering ahead of the regulatory track that phase actually requires.

---

## 15. Current Engineering Status

This charter sits alongside an active, file-by-file engineering build. As of this document's writing, the build has completed and human-tested (or human-acknowledged) Phases 1 through 7 of a 17-phase plan: configuration, macro calendar, market/global data fetching, corporate event fetching, news/sentiment fetching, event scope classification, and the global risk monitor. Remaining phases cover feature engineering, model training, ensembling, runtime validation, the core predictor, human insight management, history storage, backtesting, scheduling, and the trading dashboard itself. This charter and that build are meant to stay in sync — if one changes in a way that contradicts the other, that's a signal to update this document deliberately, not to let drift happen silently.

---

## 16. Open Items & Decisions Needed

Tracked honestly rather than glossed over:

1. **Legal review of Section 9 is not yet done.** This document reflects our best understanding of public SEBI circulars, not a lawyer's opinion. Engaging SEBI-registered securities counsel is a near-term priority, not a someday item.
2. **Pricing in Section 10 is unvalidated** — needs real customer discovery before being treated as final.
3. **Sector mapping and the NIFTY50/500 constituent lists** used in the engineering build need periodic verification against NSE's own published index composition, since these are reviewed and can change semi-annually.
4. **Festive-window and monsoon calendar dates** are approximate/illustrative in the current build and need a maintained update cadence.
5. **Naming and trademark posture** relative to the parent Alkame Inc brand should be formally confirmed in writing between the India arm and the parent entity, even though this document proceeds on the basis that this is an authorized, official use of the Alkame name.

---

## 17. Sources

- SEBI circular SEBI/HO/MIRSD/MIRSD-PoD/P/2025/0000013 (February 4, 2025), "Safer participation of retail investors in Algorithmic trading," and subsequent NSE implementation circulars.
- SEBI (Research Analysts) Regulations, 2014, and related NISM-Series-XV certification updates (effective January 2026).
- SEBI Investor Survey 2025 (public awareness vs. participation figures).
- NSE unique registered investor and demat account statistics, as reported January–May 2026 (NSE, CDSL, NSDL data, via multiple financial press summaries).
- SEBI research on individual trader losses in the derivatives segment, FY25.
- Alkame Inc public website (alkameinc.info) — Home, How It Works, Platform, About, Blog, and Careers pages, as published mid-2026.

*(Figures cited from third-party sources are paraphrased summaries for planning purposes and should be re-verified against primary sources — SEBI's own website and official exchange data — before being used in any external-facing or regulatory-filing context.)*
