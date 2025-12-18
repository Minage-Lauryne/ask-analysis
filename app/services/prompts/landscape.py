"""
Landscape Analysis Prompts
"""

from .analysis import OUTPUT_FORMAT_NOTICE, get_example_citations, CONTINUE_ANALYSIS_PROMPT

START_LANDSCAPE_PROMPT = f"""
## Landscape Analysis

Create a **Landscape Analysis** for the issue area and region as it relates to the organization's proposed work.

---

### **Purpose**
To understand the ecosystem of organizations, programs, and intermediaries operating within the defined issue area and region—identifying comparables, gaps, opportunities, and positioning options for the organization being analyzed.

---

### **Output Structure**

**Executive Summary**  
Provide a brief overview of the landscape, including:
- The service/problem domain and target population
- What is in and out of scope for this analysis (e.g., state vs. national intermediaries)
- Key themes and patterns observed across the ecosystem

**Major Organizations**  
Profile **3 leading or well-established intermediaries or program providers** with measurable statewide or national impact. For each organization, provide a structured profile that covers:
- **Organization name and mission**
- **Intervention type and approach**
- **Scale, reach, and geographic coverage**
- **Target population served**
- **Outcomes and impact** (if available)
- **Funding mix and sustainability model**
- **Key collaborations and partnerships**
- **What makes them significant in the landscape**

**REQUIRED FORMAT:** Each profile must follow this exact structure:
1. **Organization name as a subheading** (e.g., ### Organization Name)
2. **Brief introductory sentence** about the organization
3. **Bolded labels on separate lines** - Each element above must use the format "**Label:**" followed by description on the same line, then start a new line for the next label (e.g., "**Mission:**" followed by description, then "**Scale:**" on a new line with its description, etc.)
4. **Concluding sentence** about the organization's significance in the landscape
5. **Include cited sources** with links

**CRITICAL:** Do NOT write in flowing paragraph format. Each bolded label must be on its own separate line.

⚙️ *Note: When focusing on state-level intermediaries, exclude national organizations unless they have a significant state footprint.*

**Emerging or Lesser-Known Organizations**  
Profile **3 innovative or locally focused efforts** that show potential or unique models. For each organization, use the same structured format as above, emphasizing:
- What makes them innovative or promising
- How they differ from established players
- Their growth trajectory or potential

**REQUIRED FORMAT:** Follow the exact same structure as Major Organizations.

**Additional Organizations**  
List **10 additional relevant actors** with one-sentence descriptions that capture their core focus and relevance to the landscape.

**Position in the Landscape**  
Provide a 3-sentence introduction that describes how the analyzed organization fits within this ecosystem, including its potential role and strategic positioning. Follow with 3-5 bullet points that address the organization's relevance in relation to the organizations mapped above:
- How the organization's approach compares or contrasts with major players
- Unique value or positioning relative to emerging organizations
- Areas of potential overlap or complementarity
- Differentiation opportunities in the current landscape
- Strategic advantages or gaps that could be leveraged

**Gap & Opportunity Analysis**  
Assess:
- Geographic or service gaps in the current landscape
- Areas of duplication and complementarity
- Equity and access considerations
- Notable absences (e.g., if a relevant state lacks an equivalent intermediary)
- Best-practice models and evidence base, tied to local context and the organization's objectives
- State-specific innovations or differences from national trends

**Reflective Questions**  
Provide 4-6 questions to guide strategic thinking about the organization's role, such as:
- What unmet needs or gaps could the organization address?
- Which organizations represent the strongest partnership opportunities?
- How might the organization differentiate itself in this landscape?
- What lessons from established or emerging organizations could inform the organization's approach?
- Are there ecosystem dynamics or trends that the organization should consider?

**Citations**  
List all sources cited throughout the analysis with full links.

**Follow-On Prompt**  
"Would you like to extend this analysis beyond the current region (e.g., state, multi-state, or national) to identify additional comparables and innovations?"

---

### **Standards**
- Use reputable data sources (state and national reports, organization websites, credible intermediaries)
- **Cite sources and provide links** for each organization listed
- Note uncertainties and data gaps
- Adjust the number of organizations (e.g., 3/3/10) for broader analyses such as national landscapes
- Follow the structured format with bolded labels on separate lines
- Maintain a professional, analytical tone throughout

---

{get_example_citations(5)}

## Constraints

1. **Variable Handling**: If the organization name is not clearly identified in the uploaded documents, adapt the content accordingly:
   - In the "Position in the Landscape" section, use generic language like "the analyzed organization" or "the proposed organization"
   - In reflective questions, rephrase to be about "the organization" rather than using placeholders
   - Maintain all sections but ensure no placeholder text appears in the final output

2. Reply with your response in markdown format. No conversational language.

3. {OUTPUT_FORMAT_NOTICE}

4. Follow the structured format with bolded labels on separate lines as specified.

5. **Important**: After presenting all **Citations** with full links, end your response with the **Follow-On Prompt**: "Would you like to extend this analysis beyond the current region (e.g., state, multi-state, or national) to identify additional comparables and innovations?"
"""

CONTINUE_LANDSCAPE_PROMPT = f"""{CONTINUE_ANALYSIS_PROMPT}.

Your goal in this chat is to provide landscape analysis to the given documents.

{OUTPUT_FORMAT_NOTICE}"""