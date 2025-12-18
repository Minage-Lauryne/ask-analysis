"""
Leadership Analysis Prompt
"""

from .analysis import OUTPUT_FORMAT_NOTICE

LEADERSHIP_ANALYSIS_PROMPT = f"""
# Leadership & Governance Analysis

Analyze the leadership capacity, governance structure, and organizational culture based on provided documents.

## Analysis Framework

### 1. Executive Leadership Assessment
#### CEO/Executive Director Profile
- Background and relevant experience
- Leadership tenure and stability
- Public presence and thought leadership
- Succession planning (if evident)

#### Leadership Team Composition
- Team size and structure
- Diversity of backgrounds and perspectives
- Skills gap analysis
- Team cohesion indicators

#### Leadership Effectiveness Indicators
- Strategic vision and communication
- Decision-making processes
- Crisis management capability
- Innovation and adaptability

### 2. Board Governance Analysis
#### Board Composition & Structure
- Board size and member backgrounds
- Diversity metrics (gender, race, age, expertise)
- Term limits and rotation policies
- Committee structure and effectiveness

#### Board Engagement & Effectiveness
- Meeting frequency and attendance
- Strategic vs. operational focus
- Fundraising involvement
- Oversight and accountability mechanisms

#### Board Development
- Orientation and training processes
- Succession planning for board leadership
- Performance evaluation processes
- Recruitment and onboarding

### 3. Organizational Culture Assessment
#### Values & Mission Alignment
- How values manifest in daily operations
- Mission-drift indicators
- Ethical framework and decision-making
- Transparency and accountability culture

#### Staff Engagement & Development
- Employee satisfaction indicators (if available)
- Professional development opportunities
- Performance management systems
- Talent retention and turnover rates

#### Diversity, Equity & Inclusion
- DEI policies and practices
- Representation at all levels
- Inclusive decision-making processes
- Equity in compensation and advancement

### 4. Strategic Leadership Capabilities
#### Strategic Planning
- Planning process and participation
- Plan quality and implementation tracking
- Adaptability to changing conditions
- Long-term vision clarity

#### Change Management
- Past organizational change experiences
- Change management capacity
- Innovation adoption rates
- Learning organization characteristics

#### Stakeholder Management
- External relationship management
- Community engagement effectiveness
- Partner relationship quality
- Donor/funder relationship management

### 5. Risk Management & Compliance
#### Governance Risks
- Conflict of interest management
- Related party transaction oversight
- Whistleblower protections
- Document retention and transparency

#### Leadership Succession Risks
- Key person dependencies
- Succession planning adequacy
- Knowledge transfer processes
- Leadership pipeline development

#### Compliance & Ethics
- Regulatory compliance track record
- Ethical decision-making framework
- Code of conduct implementation
- Ethics training and awareness

### 6. Comparative Analysis
#### Peer Benchmarking
- Leadership team size and structure comparisons
- Board composition benchmarks
- Compensation comparability (if data available)
- Governance best practices comparison

#### Sector Standards
- Industry-specific governance requirements
- Accreditation or certification standards
- Professional association guidelines
- Regulatory requirements compliance

### 7. Key Findings & Insights
#### Strengths
- Top 3-5 leadership strengths
- Governance best practices observed
- Cultural strengths and advantages
- Leadership development successes

#### Areas for Improvement
- 3-5 priority improvement areas
- Governance gaps or weaknesses
- Cultural challenges
- Leadership development needs

#### Critical Success Factors
- What makes current leadership effective
- Key dependencies and vulnerabilities
- Essential capabilities for future success
- Culture elements that drive performance

### 8. Recommendations & Development Plan
#### Immediate Priorities (0-6 months)
1. Address critical governance gaps
2. Strengthen board committee effectiveness
3. Enhance leadership communication

#### Medium-Term Initiatives (6-18 months)
1. Leadership development programs
2. Succession planning implementation
3. Culture enhancement initiatives

#### Long-Term Strategy (18-36 months)
1. Leadership pipeline development
2. Governance model evolution
3. Culture transformation (if needed)

### 9. Measurement & Monitoring
#### Key Performance Indicators
- Board meeting effectiveness scores
- Leadership team satisfaction metrics
- Succession planning readiness scores
- DEI representation metrics

#### Monitoring Framework
- Regular assessment schedule
- Progress tracking mechanisms
- Accountability structures
- Reporting and communication plan

## Analysis Methodology

1. **Document Analysis**: Review provided materials for explicit and implicit information
2. **Pattern Recognition**: Identify recurring themes and patterns
3. **Gap Analysis**: Compare current state to best practices
4. **Risk Assessment**: Evaluate vulnerabilities and dependencies
5. **Strength Identification**: Highlight existing capabilities and advantages

## Sources of Evidence

Look for evidence in:
- Annual reports and strategic plans
- Board meeting minutes and materials
- Organizational charts and job descriptions
- Policy documents and manuals
- Staff surveys or feedback (if available)
- External evaluations or reviews
- Public communications and media

## Report Format

- Executive summary with key findings
- Clear section headers for each analysis area
- Bullet points for lists and recommendations
- Tables for comparative data
- Appendices for detailed evidence
- Action-oriented recommendations

{OUTPUT_FORMAT_NOTICE}

## Deliverable

A comprehensive 8-12 page leadership and governance analysis suitable for:
- Board self-assessment and development
- Executive coaching and development planning
- Organizational capacity building
- Funding due diligence
- Strategic planning input
"""