"""
Unified Database Module for FastAPI Agent
Supports both vector database (for RAG) and Django database (for analysis storage)
"""

from sqlalchemy import create_engine, text, exc
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
import os
from dotenv import load_dotenv
import json
from typing import Optional, Dict, Any, List
import time
import logging
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

DB_URL = os.getenv("DB_URL")  
DJANGO_DB_URL = os.getenv("DJANGO_DB_URL") or os.getenv("DATABASE_URL")  
IRS_DB_URL = os.getenv("IRS_DB_URL")  

_vector_engine = None
_django_engine = None
_irs_engine = None

_connection_stats = {
    "vector_connections": 0,
    "django_connections": 0,
    "irs_connections": 0,
    "vector_errors": 0,
    "django_errors": 0,
    "irs_errors": 0,
    "last_check": time.time()
}


def get_vector_engine():
    """Get or create vector database engine with connection pooling"""
    global _vector_engine
    
    if not DB_URL:
        logger.warning("DB_URL not set in .env - vector database disabled")
        return None
    
    if _vector_engine is None or _vector_engine.pool.status() == 'closed':
        try:
            logger.info("Creating vector database engine...")
            
            _vector_engine = create_engine(
                DB_URL,
                poolclass=QueuePool,
                pool_size=3,
                max_overflow=2,
                pool_timeout=30,
                pool_recycle=1800,
                pool_pre_ping=True,
                echo=False,
                connect_args={
                    'connect_timeout': 10,
                    'application_name': 'fastapi-agent',
                    'options': '-c statement_timeout=30000'
                }
            )
            
            with _vector_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info(" Vector DB connection established")
            _connection_stats["vector_connections"] += 1
            
        except Exception as e:
            logger.error(f" Vector DB connection failed: {e}")
            _vector_engine = None
            _connection_stats["vector_errors"] += 1
    
    return _vector_engine


def get_django_engine():
    """Get or create Django database engine for analysis storage"""
    global _django_engine
    
    if not DJANGO_DB_URL:
        logger.warning("DJANGO_DB_URL not set - Django database access disabled")
        return None
    
    if _django_engine is None or _django_engine.pool.status() == 'closed':
        try:
            logger.info("Creating Django database engine...")
            
            _django_engine = create_engine(
                DJANGO_DB_URL,
                poolclass=QueuePool,
                pool_size=3,
                max_overflow=2,
                pool_timeout=30,
                pool_recycle=1800,
                pool_pre_ping=True,
                echo=False,
                connect_args={
                    'connect_timeout': 10,
                    'application_name': 'fastapi-agent-django'
                }
            )
            
            with _django_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info(" Django DB connection established")
            _connection_stats["django_connections"] += 1
            
        except Exception as e:
            logger.error(f" Django DB connection failed: {e}")
            _django_engine = None
            _connection_stats["django_errors"] += 1
    
    return _django_engine


def get_irs_engine():
    """Get or create IRS database engine"""
    global _irs_engine
    
    if not IRS_DB_URL:
        logger.warning("IRS_DB_URL not set - IRS verification disabled")
        return None
    
    if _irs_engine is None:
        try:
            logger.info("Creating IRS database engine...")
            
            _irs_engine = create_engine(
                IRS_DB_URL,
                pool_size=2,
                max_overflow=1,
                pool_timeout=20,
                pool_recycle=1800,
                connect_args={
                    'connect_timeout': 10,
                    'application_name': 'fastapi-agent-irs'
                }
            )
            
            with _irs_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info(" IRS DB connection established")
            _connection_stats["irs_connections"] += 1
            
        except Exception as e:
            logger.error(f" IRS DB connection failed: {e}")
            _irs_engine = None
            _connection_stats["irs_errors"] += 1
    
    return _irs_engine


def get_engine(db_type: str = "vector"):
    """Get engine by type"""
    if db_type == "vector":
        return get_vector_engine()
    elif db_type == "django":
        return get_django_engine()
    elif db_type == "irs":
        return get_irs_engine()
    else:
        raise ValueError(f"Unknown database type: {db_type}")


def get_vector_session():
    """Get a database session for vector DB"""
    engine = get_vector_engine()
    if not engine:
        return None
    
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
        expire_on_commit=False
    )
    
    return SessionLocal()


def get_django_session():
    """Get a database session for Django DB"""
    engine = get_django_engine()
    if not engine:
        return None
    
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
        expire_on_commit=False
    )
    
    return SessionLocal()


def get_db():
    """Get database session for vector DB (generator)"""
    engine = get_vector_engine()
    if not engine:
        yield None
        return
    
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine
    )
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_django_db():
    """Get database connection for Django DB (generator)"""
    engine = get_django_engine()
    if not engine:
        yield None
        return
    
    conn = engine.connect()
    try:
        yield conn
    finally:
        conn.close()


def get_irs_db():
    """Get database connection for IRS DB (generator)"""
    engine = get_irs_engine()
    if not engine:
        yield None
        return
    
    conn = engine.connect()
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def get_connection(db_type: str = "vector"):
    """Context manager for database connections with timeout"""
    engine = get_engine(db_type)
    
    if not engine:
        raise ConnectionError(f"{db_type.upper()} database not configured")
    
    conn = None
    start_time = time.time()
    
    try:
        conn = engine.connect()
        
        if db_type == "vector":
            conn.execute(text("SET statement_timeout = 30000"))
        
        yield conn
        
    except exc.OperationalError as e:
        elapsed = time.time() - start_time
        logger.error(f"Database operation failed after {elapsed:.2f}s: {e}")
        raise
    except exc.TimeoutError as e:
        logger.error(f"Database timeout: {e}")
        raise
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()


def execute_query_with_timeout(
    query: str, 
    params: dict = None, 
    db_type: str = "vector", 
    timeout: int = 30
) -> Any:
    """
    Execute query with timeout and retry logic
    
    Args:
        query: SQL query
        params: Query parameters
        db_type: "vector", "django", or "irs"
        timeout: Timeout in seconds
    
    Returns:
        Query result or None
    """
    engine = get_engine(db_type)
    
    if not engine:
        logger.error(f"{db_type.upper()} database not configured")
        return None
    
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            attempt_start = time.time()
            logger.debug(f"Attempt {attempt + 1}/{max_retries + 1} - Acquiring connection...")
            
            with engine.connect() as conn:
                conn_time = time.time() - attempt_start
                logger.debug(f"Connection acquired in {conn_time:.2f}s")
                
                if db_type == "vector":
                    conn.execute(text(f"SET statement_timeout = {timeout * 1000}"))
                
                exec_start = time.time()
                result = conn.execute(text(query), params or {})
                exec_time = time.time() - exec_start
                logger.debug(f"Query executed in {exec_time:.2f}s")
                
                commit_start = time.time()
                conn.commit()
                commit_time = time.time() - commit_start
                logger.debug(f"Commit completed in {commit_time:.2f}s")
                
                if attempt > 0:
                    logger.info(f"Query succeeded on retry {attempt}")
                
                return result
                
        except exc.OperationalError as e:
            if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                logger.warning(f"Query timeout on attempt {attempt + 1}: {e}")
                if attempt < max_retries:
                    time.sleep(1)
                    continue
                else:
                    logger.error(f"Query failed after {max_retries + 1} attempts")
                    return None
            else:
                logger.error(f"Database operational error: {e}")
                return None
                
        except exc.TimeoutError as e:
            logger.error(f"Database timeout: {e}")
            return None
            
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            return None
    
    return None


def execute_query(query: str, params: dict = None, db_type: str = "vector") -> Any:
    """Execute query with default timeout (backward compatibility)"""
    return execute_query_with_timeout(query, params, db_type, timeout=30)


def get_django_single_analysis(
    analysis_id: str,
    organization_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Retrieve single analysis from Django's database
    
    Args:
        analysis_id: The analysis UUID
        organization_id: Organization identifier (for access control)
        user_id: User identifier (for access control)
    
    Returns:
        Analysis data dict or None if not found
    """
    engine = get_django_engine()
    if not engine:
        logger.error("Django database not configured")
        return None
    
    possible_tables = [
        "single_analysis_singleanalysis",
        "api_singleanalysis",
        "singleanalysis", 
    ]
    
    for table_name in possible_tables:
        try:
            query = f"""
            SELECT 
                id,
                response_text as response,
                citations,
                file_metadata,
                content_analysis,
                organization_id,
                user_id,
                chat_type,
                created_at
            FROM {table_name} 
            WHERE id = :analysis_id
              AND (:organization_id IS NULL OR organization_id = :organization_id)
              AND (:user_id IS NULL OR user_id = :user_id)
            """
            
            logger.info(f"Trying to query table: {table_name}")
            
            params = {
                "analysis_id": analysis_id,
                "organization_id": organization_id,
                "user_id": user_id
            }
            
            result = execute_query_with_timeout(
                query, 
                params, 
                db_type="django", 
                timeout=10
            )
            
            if result:
                row = result.fetchone()
                if row:
                    logger.info(f" Found analysis in table: {table_name}")
                    
                    citations = []
                    if row[2]:
                        try:
                            if isinstance(row[2], str):
                                citations = json.loads(row[2])
                            else:
                                citations = row[2]
                        except Exception as e:
                            logger.warning(f"Could not parse citations: {e}")
                    
                    file_metadata = []
                    if row[3]:
                        try:
                            if isinstance(row[3], str):
                                file_metadata = json.loads(row[3])
                            else:
                                file_metadata = row[3]
                        except Exception as e:
                            logger.warning(f"Could not parse file_metadata: {e}")
                    
                    content_analysis = {}
                    if row[4]:
                        try:
                            if isinstance(row[4], str):
                                content_analysis = json.loads(row[4])
                            else:
                                content_analysis = row[4]
                        except Exception as e:
                            logger.warning(f"Could not parse content_analysis: {e}")
                    
                    return {
                        "analysis_id": str(row[0]) if row[0] else None,
                        "response": row[1] if row[1] else "",
                        "citations": citations,
                        "file_metadata": file_metadata,
                        "content_analysis": content_analysis,
                        "organization_id": row[5] if row[5] else None,
                        "user_id": row[6] if row[6] else None,
                        "chat_type": row[7] if row[7] else "ANALYSIS",
                        "created_at": row[8].isoformat() if hasattr(row[8], 'isoformat') else str(row[8])
                    }
                
        except Exception as e:
            if "does not exist" in str(e) or "relation" in str(e):
                logger.info(f"Table {table_name} doesn't exist, trying next...")
                continue
            else:
                logger.error(f"Error querying table {table_name}: {e}")
    
    logger.error(f"Could not find analysis {analysis_id} in any Django table")
    logger.error(f"Tried tables: {possible_tables}")
    
    try:
        debug_query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        AND (table_name LIKE '%single%' OR table_name LIKE '%analysis%')
        ORDER BY table_name
        """
        
        engine = get_django_engine()
        if engine:
            with engine.connect() as conn:
                result = conn.execute(text(debug_query))
                tables = [row[0] for row in result]
                logger.info(f"Tables containing 'single' or 'analysis': {tables}")
    except Exception as e:
        logger.error(f"Debug query failed: {e}")
    
    return None


def insert_django_single_analysis(
    analysis_id: str,
    chat_type: str,
    response_text: str,
    citations: list,
    file_metadata: list,
    content_analysis: dict,
    organization_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> bool:
    """
    Insert single analysis into Django's database
    
    Returns:
        True if successful, False otherwise
    """
    engine = get_django_engine()
    if not engine:
        logger.error("Django database not configured")
        return False
    
    query = """
    INSERT INTO api_singleanalysis 
    (id, chat_type, response_text, citations, file_metadata, content_analysis, 
     organization_id, user_id, created_at)
    VALUES 
    (:id, :chat_type, :response_text, :citations, :file_metadata, :content_analysis, 
     :organization_id, :user_id, NOW())
    ON CONFLICT (id) DO UPDATE SET
        response_text = EXCLUDED.response_text,
        citations = EXCLUDED.citations,
        file_metadata = EXCLUDED.file_metadata,
        content_analysis = EXCLUDED.content_analysis,
        created_at = NOW()
    """
    
    try:
        truncated_response = response_text[:15000]
        limited_citations = citations[:50]
        
        params = {
            "id": analysis_id,
            "chat_type": chat_type,
            "response_text": truncated_response,
            "citations": json.dumps(limited_citations, default=str) if limited_citations else "[]",
            "file_metadata": json.dumps(file_metadata, default=str) if file_metadata else "[]",
            "content_analysis": json.dumps(content_analysis, default=str) if content_analysis else "{}",
            "organization_id": organization_id,
            "user_id": user_id
        }
        
        logger.info(f"Storing analysis {analysis_id} in Django database...")
        result = execute_query_with_timeout(
            query, 
            params, 
            db_type="django", 
            timeout=15
        )
        
        if result:
            logger.info(f" Analysis {analysis_id} stored in Django DB")
            return True
        else:
            logger.warning(f" Django DB storage failed for {analysis_id}")
            return False
            
    except Exception as e:
        logger.error(f" Django DB storage error for {analysis_id}: {e}")
        return False


def ensure_single_analyses_table():
    """
    Create the single_analyses table if it doesn't exist
    """
    create_table_sql = """
    CREATE SCHEMA IF NOT EXISTS agent;
    
    CREATE TABLE IF NOT EXISTS agent.single_analyses (
        id VARCHAR(255) PRIMARY KEY,
        chat_type VARCHAR(100) NOT NULL,
        response_text TEXT,
        citations JSONB,
        file_metadata JSONB,
        content_analysis JSONB,
        organization_id VARCHAR(255),
        user_id VARCHAR(255),
        created_at TIMESTAMP DEFAULT NOW()
    );
    
    CREATE INDEX IF NOT EXISTS idx_single_analyses_org_user 
    ON agent.single_analyses(organization_id, user_id);
    
    CREATE INDEX IF NOT EXISTS idx_single_analyses_created 
    ON agent.single_analyses(created_at DESC);
    """
    
    try:
        result = execute_query_with_timeout(
            create_table_sql,
            {},
            db_type="vector",
            timeout=10
        )
        logger.info(" single_analyses table ready")
        return True
    except Exception as e:
        logger.error(f" Failed to create single_analyses table: {e}")
        return False


def insert_single_analysis(
    analysis_id: str,
    chat_type: str,
    response_text: str,
    citations: list,
    file_metadata: list,
    content_analysis: dict,
    organization_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> bool:
    """
    Insert single analysis into vector database
    
    Returns:
        True if successful, False otherwise
    """
    query = """
    INSERT INTO agent.single_analyses 
    (id, chat_type, response_text, citations, file_metadata, content_analysis, 
     organization_id, user_id, created_at)
    VALUES 
    (:id, :chat_type, :response_text, CAST(:citations AS jsonb), 
     CAST(:file_metadata AS jsonb), CAST(:content_analysis AS jsonb), 
     :organization_id, :user_id, NOW())
    ON CONFLICT (id) DO UPDATE SET
        response_text = EXCLUDED.response_text,
        citations = CAST(EXCLUDED.citations AS jsonb),
        file_metadata = CAST(EXCLUDED.file_metadata AS jsonb),
        content_analysis = CAST(EXCLUDED.content_analysis AS jsonb),
        created_at = NOW()
    """
    
    try:
        truncated_response = response_text[:15000]
        limited_citations = citations[:50]
        
        params = {
            "id": analysis_id,
            "chat_type": chat_type,
            "response_text": truncated_response,
            "citations": json.dumps(limited_citations, default=str),
            "file_metadata": json.dumps(file_metadata, default=str),
            "content_analysis": json.dumps(content_analysis, default=str),
            "organization_id": organization_id,
            "user_id": user_id
        }
        
        logger.info(f"Storing analysis {analysis_id} in vector database...")
        start_time = time.time()
        
        result = execute_query_with_timeout(
            query, 
            params, 
            db_type="vector", 
            timeout=15
        )
        
        elapsed = time.time() - start_time
        
        if result:
            logger.info(f" Analysis {analysis_id} stored in vector DB (took {elapsed:.2f}s)")
            return True
        else:
            logger.warning(f" Vector DB storage failed for {analysis_id} (took {elapsed:.2f}s)")
            return False
            
    except Exception as e:
        logger.error(f" Vector DB storage error for {analysis_id}: {e}")
        return False


def get_single_analysis(
    analysis_id: str,
    organization_id: Optional[str] = None,
    user_id: Optional[str] = None,
    use_django: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Unified single analysis retrieval
    
    Args:
        analysis_id: Analysis UUID
        organization_id: Organization ID for access control
        user_id: User ID for access control
        use_django: If True, check Django DB first, else check vector DB
    
    Returns:
        Analysis data or None
    """
    if use_django:
        analysis = get_django_single_analysis(analysis_id, organization_id, user_id)
        if analysis:
            logger.info(f" Found analysis in Django DB: {analysis_id}")
            return analysis
        
        logger.info(f"Analysis not found in Django DB, checking vector DB: {analysis_id}")
    
    if organization_id and user_id:
        query = """
        SELECT id, chat_type, response_text, citations, file_metadata, 
               content_analysis, organization_id, user_id, created_at
        FROM agent.single_analyses
        WHERE id = :analysis_id
          AND organization_id = :organization_id
          AND user_id = :user_id
        """
        params = {
            "analysis_id": analysis_id,
            "organization_id": organization_id,
            "user_id": user_id
        }
    elif organization_id:
        query = """
        SELECT id, chat_type, response_text, citations, file_metadata, 
               content_analysis, organization_id, user_id, created_at
        FROM agent.single_analyses
        WHERE id = :analysis_id
          AND organization_id = :organization_id
        """
        params = {
            "analysis_id": analysis_id,
            "organization_id": organization_id
        }
    else:
        query = """
        SELECT id, chat_type, response_text, citations, file_metadata, 
               content_analysis, organization_id, user_id, created_at
        FROM agent.single_analyses
        WHERE id = :analysis_id
        """
        params = {
            "analysis_id": analysis_id
        }
    
    result = execute_query_with_timeout(query, params, db_type="vector", timeout=10)
    
    if result:
        row = result.fetchone()
        if row:
            return {
                "analysis_id": row[0],
                "chat_type": row[1],
                "response": row[2],
                "citations": row[3],
                "file_metadata": row[4],
                "content_analysis": row[5],
                "organization_id": row[6],
                "user_id": row[7],
                "created_at": row[8]
            }
    
    logger.warning(f"Analysis not found in any database: {analysis_id}")
    return None


def insert_rfp(rfp_id: str, filename: str, full_text: str, summary: dict, organization_id: Optional[str] = None, user_id: str = None):
    """Insert RFP into vector database"""
    logger.debug(f"insert_rfp called: {rfp_id}, org_id={organization_id}")
    
    query = """
    INSERT INTO agent.rfps (id, filename, full_text, summary, organization_id, user_id, created_at)
    VALUES (:id, :filename, :full_text, CAST(:summary AS jsonb), :organization_id, :user_id, NOW())
    ON CONFLICT (id) DO UPDATE SET
        filename = EXCLUDED.filename,
        full_text = EXCLUDED.full_text,
        summary = CAST(EXCLUDED.summary AS jsonb),
        organization_id = EXCLUDED.organization_id,
        user_id = EXCLUDED.user_id,
        created_at = NOW()
    """
    
    params = {
        "id": rfp_id,
        "filename": filename,
        "full_text": full_text[:100000],
        "summary": json.dumps(summary, default=str),
        "organization_id": organization_id,
        "user_id": user_id
    }
    
    result = execute_query_with_timeout(query, params, db_type="vector", timeout=10)
    
    if result:
        logger.info(f" RFP {rfp_id} stored in vector DB")
    else:
        logger.warning(f" RFP storage failed for {rfp_id}")
    
    return result


def get_rfp(rfp_id: str, organization_id: Optional[str] = None, user_id: str = None):
    """Retrieve RFP from vector database"""
    query = """
    SELECT id, filename, full_text, summary, organization_id, user_id, created_at
    FROM agent.rfps
    WHERE id = :rfp_id
      AND (
          (:organization_id IS NULL AND organization_id IS NULL) OR
          (organization_id = :organization_id)
      )
      AND user_id = :user_id
    """
    
    result = execute_query_with_timeout(
        query, 
        {
            "rfp_id": rfp_id,
            "organization_id": organization_id,
            "user_id": user_id
        }, 
        db_type="vector", 
        timeout=10
    )
    
    if result:
        row = result.fetchone()
        if row:
            return {
                "id": row[0],
                "filename": row[1],
                "full_text": row[2],
                "summary": row[3],
                "organization_id": row[4],
                "user_id": row[5],
                "created_at": row[6]
            }
    
    return None


def check_database_health() -> Dict[str, Any]:
    """
    Check health of all databases
    
    Returns:
        Health status dictionary
    """
    health = {
        "vector_db": False,
        "django_db": False,
        "irs_db": False,
        "vector_db_tables": [],
        "django_db_tables": [],
        "irs_db_tables": [],
        "connection_stats": _connection_stats.copy(),
        "timestamp": time.time()
    }
    
    vector_engine = get_vector_engine()
    if vector_engine:
        try:
            with vector_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                health["vector_db"] = True
                
                result = conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'agent'
                    ORDER BY table_name
                """))
                health["vector_db_tables"] = [row[0] for row in result]
                
        except Exception as e:
            logger.error(f"Vector DB health check failed: {e}")
            health["vector_db"] = False
    
    django_engine = get_django_engine()
    if django_engine:
        try:
            with django_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                health["django_db"] = True
                
                result = conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    AND table_name LIKE '%singleanalysis%'
                    ORDER BY table_name
                """))
                health["django_db_tables"] = [row[0] for row in result]
                
        except Exception as e:
            logger.error(f"Django DB health check failed: {e}")
            health["django_db"] = False
    
    irs_engine = get_irs_engine()
    if irs_engine:
        try:
            with irs_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                health["irs_db"] = True
                
                result = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'irs_organization_data'
                    )
                """))
                if result.fetchone()[0]:
                    health["irs_db_tables"] = ["irs_organization_data"]
                    
        except Exception as e:
            logger.error(f"IRS DB health check failed: {e}")
            health["irs_db"] = False
    
    _connection_stats["last_check"] = time.time()
    
    return health


def close_all_connections():
    """Close all database connections (call on shutdown)"""
    global _vector_engine, _django_engine, _irs_engine
    
    logger.info("Closing all database connections...")
    
    for name, engine in [
        ("Vector DB", _vector_engine),
        ("Django DB", _django_engine),
        ("IRS DB", _irs_engine)
    ]:
        if engine:
            try:
                engine.dispose()
            except Exception as e:
                logger.error(f"Error closing {name}: {e}")
    
    _vector_engine = None
    _django_engine = None
    _irs_engine = None


try:
    ensure_single_analyses_table()
except Exception as e:
    logger.warning(f"Could not initialize single_analyses table: {e}")


if __name__ == "__main__":
    """
    Test database connections
    
    Usage: python -m app.database
    """
    logger.info("Testing database connections...")
    
    health = check_database_health()
    
    logger.info("DATABASE HEALTH CHECK:")
    logger.info(f"  Vector DB: {' Connected' if health['vector_db'] else ' Failed'}")
    if health['vector_db_tables']:
        logger.info(f" Tables: {', '.join(health['vector_db_tables'])}")
    
    if health['django_db_tables']:
        logger.info(f" Tables: {', '.join(health['django_db_tables'])}")
    
    logger.info(f"  IRS DB: {' Connected' if health['irs_db'] else ' Failed'}")
    if health['irs_db_tables']:
        logger.info(f" Tables: {', '.join(health['irs_db_tables'])}")
    
    logger.info(
        f"Connection Statistics: "
        f"Vector({_connection_stats['vector_connections']}/{_connection_stats['vector_errors']}), "
        f"Django({_connection_stats['django_connections']}/{_connection_stats['django_errors']}), "
        f"IRS({_connection_stats['irs_connections']}/{_connection_stats['irs_errors']}) "
        f"[connections/errors]"
    )
    
    if health['django_db']:
        logger.info("Testing Django database access...")
        try:
            engine = get_django_engine()
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_name LIKE '%singleanalysis%'
                    LIMIT 1
                """))
                table = result.fetchone()
                if table:
                    logger.info(f"  Found Django table: {table[0]}")
                else:
                    logger.warning("  ⚠ No single analysis table found in Django DB")
        except Exception as e:
            logger.error(f"   Django DB test failed: {e}")
    
    logger.info(" Database module initialized successfully")