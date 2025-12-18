"""
Discussion Questions Prompt
"""

from .analysis import OUTPUT_FORMAT_NOTICE

DISCUSSION_QUESTIONS_PROMPT = f"""
# Discussion Questions Generator

Generate insightful, strategic discussion questions to guide deep analysis and conversation about the provided documents.

## Purpose

Create thought-provoking questions that will:
1. Uncover key insights and assumptions
2. Challenge conventional thinking
3. Explore strategic implications
4. Identify opportunities and risks
5. Guide decision-making processes

## Question Categories

Generate 3-5 questions in each of these categories:

### Category 1: Strategic Vision & Alignment
Questions about mission, vision, and long-term direction:
- How does this align with our/the organization's core mission?
- What long-term impact are we trying to achieve?
- How might this initiative evolve over 5-10 years?

### Category 2: Evidence & Impact
Questions about effectiveness and outcomes:
- What evidence supports the proposed approach?
- How will we measure real impact (not just activity)?
- What are the leading and lagging indicators of success?

### Category 3: Financial Sustainability
Questions about financial health and sustainability:
- What assumptions underpin the financial projections?
- How resilient is the funding model to market changes?
- What cost drivers have the biggest impact on sustainability?

### Category 4: Organizational Capacity
Questions about implementation capability:
- Do current capabilities match ambitious goals?
- What talent gaps need to be addressed?
- How will organizational culture support or hinder implementation?

### Category 5: Risk Assessment
Questions about potential pitfalls and challenges:
- What are the biggest assumptions that could prove wrong?
- What single point of failure would be most damaging?
- How might unintended consequences emerge?

### Category 6: Stakeholder & Community Impact
Questions about broader ecosystem effects:
- Who benefits most and who might be left behind?
- How does this affect existing community dynamics?
- What partnerships are essential for success?

### Category 7: Innovation & Learning
Questions about growth and adaptation:
- What makes this approach innovative or distinctive?
- How will we know if it's working and when to pivot?
- What lessons from similar initiatives should inform this?

### Category 8: Equity & Inclusion
Questions about fairness and representation:
- How are marginalized voices included in decision-making?
- What barriers might prevent equitable access/benefits?
- How does this address or reinforce existing inequities?

### Category 9: Implementation & Operations
Questions about practical execution:
- What's the most challenging aspect to implement?
- How will day-to-day operations need to change?
- What dependencies could derail implementation?

### Category 10: Scalability & Replication
Questions about growth potential:
- What makes this replicable in other contexts?
- What aspects might not scale well?
- How would success change the organization?

## Question Quality Guidelines

Each question should be:
1. **Open-ended**: Cannot be answered with "yes" or "no"
2. **Thought-provoking**: Encourages deeper thinking
3. **Action-oriented**: Leads to specific insights or decisions
4. **Evidence-informed**: Grounded in the document content
5. **Future-focused**: Looks ahead rather than just analyzing past
6. **Structured**: Clear, concise, and well-phrased

## Format

Organize questions by category with brief category descriptions.
For each question, include:
- The question itself
- Brief rationale (why this question matters)
- Potential follow-up probes (2-3 deeper questions)

## Example Output Structure

### Strategic Vision & Alignment
*Questions about long-term direction and mission fit*

1. **How does this initiative align with our core mission in practice, not just theory?**
   - *Rationale*: Tests whether the initiative truly advances the mission
   - *Probes*: What mission drift risks exist? How do we balance mission purity with practical needs?

2. **What would success look like 10 years from now?**
   - *Rationale*: Forces long-term thinking beyond immediate metrics
   - *Probes*: What intermediate milestones matter? How would our definition of success evolve?

## Customization

Tailor questions specifically to:
- The type of documents provided (financial, programmatic, strategic, etc.)
- The apparent stage of the organization/initiative
- The specific challenges or opportunities evident in the documents
- The decision context (funding, strategic planning, evaluation, etc.)

{OUTPUT_FORMAT_NOTICE}

## Deliverable

A comprehensive set of 30-50 discussion questions organized by category, with brief rationales and follow-up probes for each major question.
"""