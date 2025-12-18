"""
Relevant Research Prompt
"""

from .analysis import OUTPUT_FORMAT_NOTICE, get_example_citations

RELEVANT_RESEARCH_PROMPT = f"""
# Relevant Research Synthesis

Identify, analyze, and synthesize relevant research related to the organization's work, programs, or issue area.

## Research Synthesis Framework

### 1. Research Scope & Methodology
#### Scope Definition
- Key topics and research questions
- Inclusion/exclusion criteria
- Time period covered
- Geographic focus areas

#### Search Methodology
- Databases and sources consulted
- Search terms and strategies
- Quality assessment criteria
- Evidence hierarchy consideration

### 2. Key Research Themes
#### Major Finding Categories
Group research into 5-7 major thematic areas such as:
- **Effectiveness Evidence**: What works and what doesn't
- **Implementation Factors**: Critical success factors
- **Cost-Effectiveness**: Economic analysis findings
- **Equity Considerations**: Differential impacts
- **Scalability Evidence**: Replication and scale factors
- **Innovation Approaches**: Emerging methods and technologies

#### Theme Summaries
For each theme, provide:
- **Overview**: Main findings and conclusions
- **Strength of Evidence**: Quality and quantity of research
- **Consensus Level**: Agreement among studies
- **Gaps & Controversies**: Areas of uncertainty or debate

### 3. High-Impact Studies Analysis
#### Landmark Studies (5-10 studies)
For each landmark study, analyze:
- **Study Design**: Methodology and rigor
- **Key Findings**: Major results and conclusions
- **Limitations**: Study weaknesses or constraints
- **Relevance**: Application to current context
- **Citations**: Full citation with DOI/link

#### Systematic Reviews & Meta-Analyses
- Summary of major reviews
- Pooled effect sizes if available
- Quality assessment of reviews
- Implications for practice

### 4. Evidence-Based Practices
#### Proven Interventions
- Interventions with strong evidence
- Effect sizes and confidence intervals
- Implementation requirements
- Cost considerations

#### Promising Approaches
- Emerging evidence areas
- Pilot study results
- Theory-based interventions
- Adaptation considerations

#### Ineffective Approaches
- Interventions shown not to work
- Harmful or counterproductive approaches
- Common mistakes to avoid
- Lessons from failures

### 5. Contextual Factors Analysis
#### Implementation Context
- Organizational capacity requirements
- Staff training and qualifications
- Cultural adaptation needs
- Policy and regulatory environment

#### Population Considerations
- Differential effectiveness by subgroup
- Cultural appropriateness
- Accessibility and equity
- Special populations considerations

#### Geographic & Temporal Factors
- Regional variations in effectiveness
- Historical context considerations
- Changing social/economic conditions
- Future trend implications

### 6. Research Gaps & Future Directions
#### Critical Knowledge Gaps
- Areas needing more research
- Methodological limitations
- Population gaps
- Contextual gaps

#### Research Priority Areas
- Highest impact research questions
- Practical vs. theoretical priorities
- Short-term vs. long-term needs
- Funders' role in research agenda

#### Emerging Research Trends
- New methodologies
- Technology-enabled research
- Participatory research approaches
- Transdisciplinary research

### 7. Practical Applications
#### Program Design Implications
- Evidence-informed design principles
- Adaptation guidance
- Implementation checklists
- Quality assurance frameworks

#### Evaluation & Measurement
- Recommended outcome measures
- Data collection methods
- Analysis approaches
- Reporting standards

#### Policy & Funding Implications
- Policy recommendations
- Funding allocation guidance
- Accountability frameworks
- Performance-based contracting

### 8. Synthesis & Integration
#### Cross-Cutting Themes
- Common findings across studies
- Contradictory evidence reconciliation
- Evolution of knowledge over time
- Paradigm shifts or breakthroughs

#### Confidence Assessment
- Overall confidence in evidence
- Areas of strong vs. weak evidence
- Risk assessment for acting on evidence
- Decision-making under uncertainty

### 9. Recommendations for Action
#### For Practitioners
- Immediate implementation steps
- Professional development needs
- Partnership opportunities
- Resource requirements

#### For Funders
- Research funding priorities
- Program funding criteria
- Evaluation requirements
- Learning agenda development

#### For Researchers
- Priority research questions
- Methodology improvements
- Dissemination strategies
- Practice-research partnerships

### 10. Appendices & References
#### Detailed Study Summaries
- Expanded analysis of key studies
- Methodological details
- Full results reporting
- Critical appraisal scores

#### Reference Database
- Complete bibliography
- Organized by theme/topic
- Searchable format
- Link repository

{get_example_citations(15)}

## Synthesis Standards

1. **Rigorous**: Systematic approach to evidence review
2. **Comprehensive**: Cover breadth and depth of research
3. **Critical**: Evaluate study quality and limitations
4. **Practical**: Focus on actionable insights
5. **Transparent**: Clearly document methods and sources
6. **Balanced**: Present strengths and limitations of evidence

## Output Format

- Executive summary with key evidence findings
- Thematic organization with clear headers
- Study summaries in consistent format
- Evidence tables and synthesis matrices
- Practical application sections
- Complete reference list

{OUTPUT_FORMAT_NOTICE}

## Deliverable

A comprehensive research synthesis report (15-25 pages) including:
- Evidence summary by thematic area
- Analysis of key studies
- Practical application guidance
- Research gap identification
- Complete references and citations
"""