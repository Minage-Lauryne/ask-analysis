"""
Board Memo Prompts
"""

from .analysis import OUTPUT_FORMAT_NOTICE

BOARD_MEMO_PROMPT = f"""---
title: "Board Memo"
description: "A concise, neutral, and evidence-based briefing to support board discussion and collective judgment."
---

# Board Memo

A concise, neutral, and evidence-based briefing designed to support thoughtful board discussion and collective judgment.

---

## Purpose

This memo organizes key information to help the board interpret, reflect on, and discuss an organization's capacity, strategy, and potential contribution within its field.  
It is **not** a recommendation or advocacy document — it provides structure and evidence for the board's deliberation.

---

## Structure

### 1. Executive Summary (≤250 words)

Provide a brief, narrative overview highlighting the most relevant insights for board reflection.

- Core themes emerging from the analysis  
- High-level considerations or questions for discussion  
- Notable strengths, uncertainties, or contextual dynamics  

The tone should be factual and balanced, guiding the reader to understand what matters most without signaling a preferred outcome.

---

### 2. Context Snapshot

Offer a concise portrait of the organization and its operating environment.

- Mission and core activities  
- Populations or issues served  
- Broader context: trends, needs, or systemic factors shaping its work  
- How the organization positions itself within that landscape  

Narrative should integrate uploaded evidence and cite sources where applicable.

---

### 3. Evidence Overview

Summarize what credible research and practice say about the organization's general approach or strategy area.

- Major findings from peer-reviewed studies, evaluations, or field literature  
- Consistency (or divergence) between research insights and the organization's methods  
- Gaps, uncertainties, or emerging debates in the evidence base  

Include concise **in-text citations** (author, year) and a short reference list with links.

---

### 4. Organizational Readiness

Describe the organization's internal capacity to execute and adapt effectively.

- Leadership and governance quality  
- Staffing depth and expertise  
- Operational systems, culture, and adaptability  
- Strengths and challenges indicated by the available documentation  

Integrate evidence from uploaded materials; note where information is incomplete or ambiguous.

---

### 5. Financial Overview

Provide a clear summary of the organization's financial condition and trends.

- Liquidity, reserves, and solvency indicators  
- Revenue diversity and predictability  
- Cost structure and spending priorities  
- Observed patterns across recent fiscal periods  

Present findings factually, referencing nonprofit finance standards without interpretation or judgment.

---

### 6. Equity and Inclusion Reflection

Highlight how equity and representation are addressed in both the organization's structure and its work.

- Demographics and lived experience in leadership and staff  
- Inclusion of community voice and power in decision-making  
- Accessibility and responsiveness to the populations served  
- Reflection on assumptions or biases that may shape analysis or practice  

---

### 7. Benefit–Cost View

Provide an interpretive summary (not a valuation) of how the organization's efforts relate to the scale of investment and anticipated benefit.

- Key categories of benefit (quantitative or qualitative)
- Cost considerations or operational tradeoffs  
- Noted sensitivities or contextual dependencies  

This section should illuminate relationships between effort, cost, and potential impact without implying directionality.

---

### 8. Implementation Factors

Identify practical and contextual considerations relevant to ongoing monitoring or support.

- Major operational or environmental risks  
- Early indicators of progress or challenge  
- Organizational responses or mitigations observed or documented  

---

### 9. Learning Agenda

Outline key questions or areas of uncertainty that warrant observation or inquiry over time.

- What can be learned from this organization's approach or experience?  
- Which outcomes or conditions will be most informative to track?  
- How might insights from this case inform broader strategy or policy learning?  

---

## Standards for Preparation

- Maintain a **neutral and evidence-based** tone throughout  
- Support every factual statement with a **citation** or clear attribution to uploaded data or credible research  
- Distinguish among **facts, interpretations, and assumptions**  
- Identify **data gaps** and describe how they affect confidence in analysis  
- Write in **succinct, readable paragraphs** supported by **bulleted points** for clarity  

---

{OUTPUT_FORMAT_NOTICE}
"""