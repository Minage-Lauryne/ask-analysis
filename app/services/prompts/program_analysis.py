"""
Program Analysis Prompt
"""

from .analysis import OUTPUT_FORMAT_NOTICE, get_example_citations

PROGRAM_ANALYSIS_PROMPT = f"""
# Comprehensive Program Analysis

Analyze the effectiveness, efficiency, and impact of organizational programs based on provided documents.

## Analysis Framework

### 1. Program Portfolio Overview
#### Program Inventory
- List all programs/services with brief descriptions
- Program age and evolution history
- Target populations served
- Geographic reach and scale

#### Program Categorization
- Core vs. peripheral programs
- Revenue-generating vs. mission-focused
- Established vs. pilot programs
- Direct service vs. capacity building

### 2. Program Design & Theory of Change
#### Program Logic Models
- Inputs, activities, outputs, outcomes, impact
- Assumptions and external factors
- Causal pathways and logic

#### Evidence Base
- Research supporting program approach
- Evidence gaps or uncertainties
- Adaptation to local context
- Innovation elements

#### Target Population Analysis
- Needs assessment alignment
- Demographic characteristics
- Access and equity considerations
- Cultural responsiveness

### 3. Implementation Analysis
#### Program Delivery
- Delivery methods and modalities
- Staffing and volunteer models
- Partner roles and relationships
- Facilities and equipment needs

#### Quality Assurance
- Program fidelity measures
- Quality control processes
- Staff training and supervision
- Participant feedback mechanisms

#### Adaptation & Learning
- Program adaptation history
- Learning from implementation
- Continuous improvement processes
- Innovation incorporation

### 4. Outcomes & Impact Assessment
#### Outcome Measurement
- Output and outcome metrics
- Data collection methods
- Data quality and reliability
- Participant tracking systems

#### Impact Evidence
- Demonstrated outcomes and results
- Comparative effectiveness data
- Long-term impact indicators
- Cost-effectiveness evidence

#### Participant Outcomes
- Immediate participant benefits
- Intermediate outcomes
- Long-term impact on participants
- Unintended consequences

### 5. Financial Analysis
#### Program Economics
- Cost per participant/outcome
- Revenue generation vs. costs
- Subsidy requirements
- Economies of scale analysis

#### Funding Analysis
- Program-specific funding sources
- Grant dependency levels
- Fee-for-service viability
- Cross-subsidization patterns

#### Efficiency Metrics
- Administrative cost ratios
- Fundraising efficiency
- Resource utilization rates
- Overhead allocation fairness

### 6. Capacity & Scalability
#### Current Capacity Assessment
- Program at capacity indicators
- Waitlists or unmet demand
- Staffing adequacy
- Facility constraints

#### Scalability Potential
- Replication feasibility
- Expansion barriers
- Technology enablement
- Partnership requirements

#### Sustainability Analysis
- Long-term funding prospects
- Community support levels
- Policy environment factors
- Competitive positioning

### 7. Risk Assessment
#### Implementation Risks
- Staff turnover risks
- Funding instability
- Regulatory compliance
- Quality maintenance

#### External Risks
- Market/demand changes
- Competitive pressures
- Policy/regulatory changes
- Economic environment shifts

#### Mitigation Strategies
- Risk management plans
- Contingency planning
- Insurance and protections
- Diversification strategies

### 8. Comparative Analysis
#### Peer Benchmarking
- Similar program comparisons
- Best practice identification
- Innovation assessment
- Cost-effectiveness comparisons

#### Sector Standards
- Industry standards compliance
- Accreditation requirements
- Professional guidelines
- Ethical standards adherence

### 9. Strengths, Weaknesses, Opportunities, Threats (SWOT)
#### Program-Level SWOT
- Internal strengths and weaknesses
- External opportunities and threats
- Strategic implications
- Priority action areas

### 10. Recommendations & Improvement Plan
#### Strategic Recommendations
1. **Program Enhancement**: Quality improvements
2. **Efficiency Gains**: Cost reduction opportunities
3. **Impact Expansion**: Scale or reach expansion
4. **Innovation Development**: New approaches or technologies

#### Implementation Priorities
- Immediate actions (0-3 months)
- Short-term improvements (3-12 months)
- Long-term developments (1-3 years)

#### Resource Requirements
- Staffing needs
- Funding requirements
- Technology investments
- Partnership development

### 11. Monitoring & Evaluation Framework
#### Performance Indicators
- Output and outcome metrics
- Efficiency measures
- Quality indicators
- Participant satisfaction

#### Evaluation Plan
- Regular assessment schedule
- Data collection methods
- Analysis and reporting
- Learning incorporation

## Analysis Standards

1. **Evidence-Based**: Ground conclusions in data
2. **Participant-Centered**: Focus on participant impact
3. **Practical**: Actionable recommendations
4. **Comprehensive**: Cover all program dimensions
5. **Comparative**: Use benchmarks and standards

{get_example_citations(8)}

## Data Requirements

Analyze available data on:
- Program participation numbers and demographics
- Outcome measurement results
- Financial performance data
- Staffing and operational metrics
- Participant feedback and satisfaction
- External evaluations or studies

## Report Format

- Executive summary with key findings
- Clear section organization
- Data tables and visualizations where helpful
- Bulleted recommendations
- Appendices for detailed data
- Implementation roadmap

{OUTPUT_FORMAT_NOTICE}

## Deliverable

A comprehensive 10-15 page program analysis report suitable for:
- Program improvement planning
- Funding proposals and reports
- Strategic decision-making
- Board oversight and governance
- Staff development and training
"""