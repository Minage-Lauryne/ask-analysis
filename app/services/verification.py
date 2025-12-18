import os
import logging
import requests
from typing import Optional, Dict, Any, Tuple
from sqlalchemy import create_engine, text
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

IRS_DB_URL = os.getenv("IRS_DB_URL")

if not IRS_DB_URL:
    logger.warning("IRS_DB_URL not set in environment - verification will be limited")
    irs_engine = None
else:
    try:
        irs_engine = create_engine(IRS_DB_URL)
        with irs_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("IRS database connection established")
    except Exception as e:
        logger.error(f"IRS database connection failed: {e}", exc_info=True)
        irs_engine = None


def extract_organization_name(proposal_text: str) -> Tuple[str, float]:
    """
    Extract organization name from proposal text using AI.
    
    Args:
        proposal_text: Full text of the proposal document
    
    Returns:
        Tuple of (organization_name, confidence_score)
        confidence_score ranges from 0.0 to 1.0
    """
    from app.services.comparative_analysis import safe_generate
    
    system_msg = "You extract organization names from grant proposals. Return only the name, nothing else."
    
    user_msg = f"""Extract the exact organization name from this proposal.

Return just the organization name with no punctuation at the end.


Proposal text (first 250 characters):
{proposal_text[:250]}

Organization name:"""
    
    try:
        result = safe_generate(system_msg, user_msg, max_tokens=50)
        
        if result and len(result.strip()) > 3:
            org_name = result.strip()
            org_name = org_name.strip('"\'.,;:#*[](){}')
            
            if len(org_name) < 3 or len(org_name) > 150:
                raise ValueError(f"Name length invalid: {len(org_name)} characters")
            
            confidence = 0.9 if len(org_name) > 15 else 0.75
            
            return org_name, confidence
            
    except Exception as e:
        logger.debug(f"AI extraction failed: {e}")
    
    for line in proposal_text.split('\n')[:30]:
        line = line.strip().lstrip('#*-• ').strip()
        if (10 < len(line) < 100 and 
            line[0].isupper() and 
            not line.lower().startswith(('summary', 'introduction', 'proposal', 'background'))):
            logger.debug(f"Fallback to first line: '{line}' (confidence: 0.3)")
            return line, 0.3
    
    logger.warning("Could not extract organization name (confidence: 0.1)")
    return "Unknown Organization", 0.1


def query_irs_database(org_name: str) -> Tuple[Optional[str], Optional[str], float]:
    if not irs_engine:
        logger.debug("IRS database not available")
        return None, None, 0.0
    
    try:
        with irs_engine.connect() as conn:
            exact_query = text("""
                SELECT ein, organization_name
                FROM irs_organization_data
                WHERE LOWER(TRIM(organization_name)) = LOWER(TRIM(:org_name))
                LIMIT 1
            """)
            
            result = conn.execute(exact_query, {"org_name": org_name}).fetchone()
            
            if result:
                ein, legal_name = result
                return ein, legal_name, 1.0
            
            fuzzy_query = text("""
                SELECT 
                    ein, 
                    organization_name,
                    similarity(LOWER(organization_name), LOWER(:org_name)) as score
                FROM irs_organization_data
                WHERE similarity(LOWER(organization_name), LOWER(:org_name)) > 0.3
                ORDER BY score DESC
                LIMIT 1
            """)
            
            result = conn.execute(fuzzy_query, {"org_name": org_name}).fetchone()
            
            if result:
                ein, legal_name, similarity_score = result
                
                if similarity_score >= 0.6:
                    return ein, legal_name, float(similarity_score)
                else:
                    logger.info(f"NO RELIABLE MATCH for: '{org_name}'")
                    return None, None, 0.0
            
            logger.info(f"NO IRS MATCH: '{org_name}'")
            return None, None, 0.0
            
    except Exception as e:
        logger.error(f"IRS query error: {e}", exc_info=True)
        return None, None, 0.0


def call_propublica_api(ein: str) -> Dict[str, Any]:
    """
    Retrieve nonprofit financial data from ProPublica Nonprofit Explorer API.
    
    API Documentation: https://projects.propublica.org/nonprofits/api/
    
    Args:
        ein: Tax ID (e.g., "12-3456789")
    
    Returns:
        Dictionary containing:
        - revenue: Total income (int)
        - assets: Total assets (int)
        - tax_status: IRS subsection (e.g., "501(c)(3)")
        - filing_status: "Active" or "Revoked"
        - year: Tax period year (str)
        
        Returns empty dict if API call fails.
    """
    try:
        url = f"https://projects.propublica.org/nonprofits/api/v2/organizations/{ein}.json"
        
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            org = data.get('organization', {})
            
            revenue = org.get('income', 0)
            assets = org.get('assets', 0)
            
            result = {
                "revenue": revenue if isinstance(revenue, (int, float)) else 0,
                "assets": assets if isinstance(assets, (int, float)) else 0,
                "tax_status": org.get('subsection', 'Unknown'),
                "filing_status": "Active" if org.get('revocation_status') is None else "Revoked",
                "year": str(org.get('tax_period', 'Unknown'))
            }
            
            logger.info(f"ProPublica data: Revenue ${result['revenue']:,}, Assets ${result['assets']:,}")
            return result
            
        elif response.status_code == 404:
            logger.debug(f"ProPublica: No data found for EIN {ein}")
            return {}
        else:
            logger.warning(f"ProPublica API returned status {response.status_code}")
            return {}
            
    except requests.exceptions.Timeout:
        logger.warning("ProPublica API timeout")
        return {}
    except Exception as e:
        logger.error(f"ProPublica API error: {e}")
        return {}


def check_sanctions(org_name: str, ein: Optional[str] = None) -> Dict[str, Any]:
    
    sanctions_api_key = os.getenv("SANCTIONS_API_KEY")
    
    if not sanctions_api_key:
        logger.warning("SANCTIONS_API_KEY not set in environment - skipping sanctions check")
        return {
            "checked": False,
            "match_found": None,
            "error": "SANCTIONS_API_KEY not configured in environment"
        }
    
    try:
        url = "https://data.trade.gov/consolidated_screening_list/v1/search"
        
        headers = {
            "subscription-key": sanctions_api_key
        }
        
        params = {
            "name": org_name,
           
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=15)
        
        logger.debug(f"Sanctions API Status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                results = data.get('results', [])
                
                confirmed_matches = []
                for result in results:
                    result_name = result.get('name', '').lower()
                    query_name = org_name.lower()
                    
                    if result_name in query_name or query_name in result_name:
                        confirmed_matches.append(result)
                
                match_found = len(confirmed_matches) > 0
                
                if match_found:
                    logger.warning(f"SANCTION MATCH FOUND: {len(confirmed_matches)} result(s)")
                    for match in confirmed_matches[:2]:
                        logger.warning(f"  - {match.get('name')} ({match.get('source')})")
                else:
                    logger.info("SANCTIONS CHECK CLEAR")
                
                return {
                    "checked": True,
                    "match_found": match_found,
                    "confidence": 1.0 if match_found else 0.0,
                    "lists_checked": list(set([r.get('source', 'Unknown') for r in results])),
                    "details": confirmed_matches[:3]
                }
            except ValueError as e:
                logger.error(f"Sanctions API JSON decode error: {e}")
                return {
                    "checked": False,
                    "match_found": None,
                    "error": f"JSON decode error: {str(e)}"
                }
            
        elif response.status_code == 401:
            logger.error("Sanctions API authentication failed - check SANCTIONS_API_KEY")
            return {
                "checked": False,
                "match_found": None,
                "error": "Authentication failed - check API key in environment"
            }
        else:
            logger.warning(f"Sanctions API returned status {response.status_code}")
            return {
                "checked": False,
                "match_found": None,
                "error": f"API returned status {response.status_code}"
            }
            
    except requests.exceptions.Timeout:
        logger.warning("Sanctions API timeout")
        return {
            "checked": False,
            "match_found": None,
            "error": "API timeout"
        }
    except Exception as e:
        logger.error(f"Sanctions check error: {e}")
        return {
            "checked": False,
            "match_found": None,
            "error": str(e)
        }

def calculate_risk_level(
    verified: bool,
    irs_confidence: float,
    sanction_data: Dict[str, Any],
    financial_data: Dict[str, Any]
) -> str:
   
    if sanction_data.get("match_found"):
        return "CRITICAL"
    
    if not verified:
        return "HIGH"
    
    if irs_confidence < 0.7:
        return "HIGH"
    
    revenue = financial_data.get("revenue", 0)
    if isinstance(revenue, (int, float)) and revenue < 50000:
        return "MEDIUM"
    
    if irs_confidence < 0.85:
        return "MEDIUM"
    
    return "LOW"


def verify_organization(proposal_text: str) -> Dict[str, Any]:
    
    logger.info("ORGANIZATION VERIFICATION PIPELINE")
    
    org_name, name_confidence = extract_organization_name(proposal_text)
    
    ein, legal_name, irs_confidence = query_irs_database(org_name)
    verified = ein is not None
    
    if ein:
        financial_data = call_propublica_api(ein)
    else:
        logger.debug("Skipping ProPublica (no EIN found)")
        financial_data = {}
    
    logger.info("[4/4] Checking sanctions lists...")
    sanction_data = check_sanctions(org_name, ein)
    
    risk_level = calculate_risk_level(verified, irs_confidence, sanction_data, financial_data)
    
    verification_result = {
        "org_name": org_name,
        "name_confidence": round(name_confidence, 2),
        "verified": verified,
        "ein": ein,
        "legal_name": legal_name or org_name,
        "irs_confidence": round(irs_confidence, 2),
        "revenue": financial_data.get("revenue"),
        "assets": financial_data.get("assets"),
        "tax_status": financial_data.get("tax_status", "Unknown"),
        "filing_status": financial_data.get("filing_status", "Unknown"),
        "tax_year": financial_data.get("year", "Unknown"),
        "sanction_checked": sanction_data.get("checked", False),
        "sanction_clear": not sanction_data.get("match_found", False) if sanction_data.get("checked") else None,
        "sanction_details": sanction_data.get("details", []),
        "sanction_lists": sanction_data.get("lists_checked", []),
        "risk_level": risk_level,
        "verified_at": datetime.now().isoformat()
    }
    
    logger.info("VERIFICATION COMPLETE")
    if ein:
        logger.info(f"EIN: {ein}")
    if financial_data:
        logger.info(f"Revenue: ${financial_data.get('revenue', 0):,}")
        logger.info(f"Assets: ${financial_data.get('assets', 0):,}")
    sanction_status = 'CLEAR' if verification_result['sanction_clear'] else 'MATCH FOUND' if sanction_data.get('match_found') else 'NOT CHECKED'
    logger.info(f"Sanctions: {sanction_status}")
    logger.info(f"Risk Level: {risk_level}")
    
    return verification_result


def build_verification_context(verification_data: Dict[str, Any]) -> str:
    verified = verification_data['verified']
    ein = verification_data.get('ein', 'Not Found')
    legal_name = verification_data['legal_name']
    irs_conf = verification_data['irs_confidence']
    
    revenue = verification_data.get('revenue')
    assets = verification_data.get('assets')
    tax_status = verification_data.get('tax_status', 'Unknown')
    filing_status = verification_data.get('filing_status', 'Unknown')
    
    sanction_checked = verification_data.get('sanction_checked', False)
    sanction_clear = verification_data.get('sanction_clear')
    risk_level = verification_data['risk_level']
    
    revenue_str = f"${revenue:,}" if isinstance(revenue, (int, float)) else "Unknown"
    assets_str = f"${assets:,}" if isinstance(assets, (int, float)) else "Unknown"
    
    if not sanction_checked:
        sanction_str = "NOT CHECKED"
    elif sanction_clear:
        sanction_str = "CLEAR (no matches found)"
    else:
        sanction_str = "MATCH FOUND (CRITICAL)"
    
    context = f"""
VERIFIED ORGANIZATION DATA (Use in WHO analysis)

Organization: {legal_name}
IRS Status: {'VERIFIED' if verified else 'UNVERIFIED'}
EIN: {ein}
IRS Match Confidence: {irs_conf:.0%}

Financial Health (most recent filing):
  - Revenue: {revenue_str}
  - Assets: {assets_str}
  - Tax Status: {tax_status}
  - Filing Status: {filing_status}

Compliance and Risk:
  - Sanctions Screening: {sanction_str}
  - Overall Risk Level: {risk_level}

INSTRUCTIONS FOR WHO ANALYSIS:
You must incorporate this verified data into your WHO assessment.
Discuss:
  1. Whether the organization is IRS-verified and implications
  2. Financial capacity based on revenue and assets
  3. Compliance status (sanctions clear or not)
  4. Overall risk level and its significance
  5. Whether they have sufficient organizational capacity to execute

If risk level is HIGH or CRITICAL, this should factor heavily into your
recommendation, regardless of proposal content quality.
"""
    
    return context