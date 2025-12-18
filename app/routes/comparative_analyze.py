from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional, List
from app.services.comparative_analysis import (
    analyze_uploaded_files,
    analyze_with_instructions,
    analyze_with_evidence_only,
)
import uuid
import logging
import traceback


router = APIRouter()

AGENT_DB_STORAGE_ENABLED = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@router.post("/")
async def analyze_proposals(
    proposals: List[UploadFile] = File(...),
    rfp: Optional[UploadFile] = File(None),
    instructions: Optional[str] = Form(None),
    organization_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    submission_id: Optional[str] = Form(None),
):
    try:

        if rfp:
            content = await rfp.read(10)
            await rfp.seek(0)
            if not rfp.filename or rfp.filename.strip() == "" or len(content) == 0:
                logger.debug(f"Empty rfp detected, treating as None")
                rfp = None
        logger.info(f"RFP: {'Provided' if rfp else 'Not provided'}")

        if not proposals:
            raise HTTPException(
                status_code=400, detail="At least one proposal file is required"
            )

        if rfp and instructions:
            logger.debug(
                f"Both RFP and instructions provided - using RFP, ignoring instructions"
            )
            instructions = None

        analysis_id = submission_id if submission_id else str(uuid.uuid4())
        logger.debug(f"Final analysis ID: {analysis_id}")

        if rfp:
            logger.info(f"RFP-based analysis with {len(proposals)} proposals")
            result = await analyze_uploaded_files(
                rfp,
                proposals,
                organization_id=organization_id,
                user_id=user_id,
                submission_id=submission_id,
            )
        elif instructions:
            logger.info(
                f"[analyze] Instruction-based analysis with {len(proposals)} proposals"
            )
            result = await analyze_with_instructions(
                instructions,
                proposals,
                organization_id=organization_id,
                user_id=user_id,
                submission_id=submission_id,
            )
        else:
            logger.info(
                f"Evidence-based analysis (no RFP or instructions) with {len(proposals)} proposals"
            )
            result = await analyze_with_evidence_only(
                proposals,
                organization_id=organization_id,
                user_id=user_id,
                submission_id=submission_id,
            )

        if submission_id and result.get("rfp_id") != submission_id:
            logger.warning(
                f"ID mismatch - Django sent {submission_id}, but agent returned {result.get('rfp_id')}"
            )
            result["rfp_id"] = submission_id

        logger.debug(f"Final result using ID: {result['rfp_id']}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
