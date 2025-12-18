from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os
import logging
from dotenv import load_dotenv
import json
from typing import Optional

load_dotenv()

logger = logging.getLogger(__name__)

DB_URL = os.getenv("DB_URL")

if not DB_URL:
    logger.warning("DB_URL not set in .env - database functionality disabled")
    engine = None
    SessionLocal = None
else:
    try:
        engine = create_engine(DB_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("PostgreSQL connection established")
    except Exception as e:
        logger.error(f"PostgreSQL connection failed: {e}", exc_info=True)
        engine = None
        SessionLocal = None


def get_db():
    """Get database session"""
    if SessionLocal:
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
    else:
        yield None


def execute_query(query: str, params: dict = None):
    """Execute a SQL query"""
    if not engine:
        logger.error("Database not configured - cannot execute query")
        return None
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            conn.commit()
            return result
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        return None


def get_storage_filename_from_paper_id(paper_id: str) -> Optional[str]:
    """
    Look up the actual storage filename from paper_id using research_corpus table
    
    Args:
        paper_id: The paper ID from research_chunks
        
    Returns:
        The storage filename (paper_id.pdf) or None if not found
    """
    if not engine or not paper_id:
        return None
    
    try:
        query = """
        SELECT paper_id 
        FROM agent.research_corpus 
        WHERE paper_id = :paper_id 
        LIMIT 1
        """
        
        with engine.connect() as conn:
            result = conn.execute(text(query), {"paper_id": paper_id})
            row = result.fetchone()
            
            if row:
                return f"{paper_id}.pdf"
            else:
                return None
                
    except Exception as e:
        logger.error(f"Error looking up paper_id {paper_id}: {e}", exc_info=True)
        return None


def insert_rfp(rfp_id: str, filename: str, full_text: str, summary: dict):
    """Insert RFP into agent.rfps table"""
    query = """
    INSERT INTO agent.rfps (id, filename, full_text, summary, created_at)
    VALUES (:id, :filename, :full_text, :summary::jsonb, NOW())
    """
    
    params = {
        "id": rfp_id,
        "filename": filename,
        "full_text": full_text,
        "summary": json.dumps(summary)
    }
    
    return execute_query(query, params)


def insert_proposal(proposal_id: str, rfp_id: str, filename: str, full_text: str, 
                   org_name: str, recommendation: str, score: int, budget: str, 
                   timeline: str, analysis: dict, verification: dict = None):
    """Insert proposal into agent.proposals table"""
    query = """
    INSERT INTO agent.proposals 
    (id, rfp_id, filename, full_text, organization_name, recommendation, 
     score, budget, timeline, analysis, verification, created_at)
    VALUES 
    (:id, :rfp_id, :filename, :full_text, :org_name, :recommendation,
     :score, :budget, :timeline, :analysis::jsonb, :verification::jsonb, NOW())
    """
    
    params = {
        "id": proposal_id,
        "rfp_id": rfp_id,
        "filename": filename,
        "full_text": full_text,
        "org_name": org_name,
        "recommendation": recommendation,
        "score": score,
        "budget": budget,
        "timeline": timeline,
        "analysis": json.dumps(analysis),
        "verification": json.dumps(verification) if verification else None
    }
    
    return execute_query(query, params)