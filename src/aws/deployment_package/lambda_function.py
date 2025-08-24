"""
AWS Lambda Function - Clinical Trial Safety Event Processor
Main entry point for Lambda deployment
"""

import json
import os
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """
    AWS Lambda entry point - processes adverse events
    
    Args:
        event: Event data (adverse event information)
        context: Lambda runtime context
        
    Returns:
        Processing results
    """
    try:
        logger.info(f"🚀 Lambda invoked at {datetime.utcnow()}")
        logger.info(f"📨 Event received: {json.dumps(event, default=str)[:200]}...")
        
        # Import processing logic (lazy loading for cold start optimization)
        from lambda_handler import SafetyEventProcessor
        
        # Initialize processor
        processor = SafetyEventProcessor()
        
        # Process the event
        result = processor.process_event(event)
        
        # Return success response
        response = {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': True,
                'message': 'Event processed successfully',
                'result': result,
                'lambda_request_id': context.aws_request_id,
                'timestamp': datetime.utcnow().isoformat()
            })
        }
        
        logger.info("✅ Lambda processing completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"❌ Lambda error: {str(e)}")
        
        # Return error response
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': False,
                'error': str(e),
                'message': 'Event processing failed',
                'lambda_request_id': context.aws_request_id if context else 'unknown',
                'timestamp': datetime.utcnow().isoformat()
            })
        }