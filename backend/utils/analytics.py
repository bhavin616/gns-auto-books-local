"""
Analytics Utility

Centralized analytics event logging system for tracking user and client activities.
"""

from typing import Optional, Dict, Any
from datetime import datetime
from bson import ObjectId
from backend.services.mongodb import get_database
from backend.utils.logger import log_message


def get_device_info(request: Any) -> Dict[str, Any]:
    """
    Extract device information from request headers.
    
    Args:
        request: FastAPI/Starlette request object
        
    Returns:
        Dict containing device, browser, and OS information
    """
    user_agent = request.headers.get('user-agent', '') if request else ''
    device_info = {
        'userAgent': user_agent
    }
    
    if user_agent:
        user_agent_lower = user_agent.lower()
        
        # Device detection
        if any(term in user_agent_lower for term in ['mobile', 'android', 'iphone', 'ipad']):
            if any(term in user_agent_lower for term in ['tablet', 'ipad']):
                device_info['device'] = 'Tablet'
            else:
                device_info['device'] = 'Mobile'
        else:
            device_info['device'] = 'Desktop'
        
        # Browser detection
        if 'chrome' in user_agent_lower and not any(term in user_agent_lower for term in ['edge', 'edg']):
            device_info['browser'] = 'Chrome'
        elif 'firefox' in user_agent_lower:
            device_info['browser'] = 'Firefox'
        elif 'safari' in user_agent_lower and 'chrome' not in user_agent_lower:
            device_info['browser'] = 'Safari'
        elif any(term in user_agent_lower for term in ['edge', 'edg']):
            device_info['browser'] = 'Edge'
        else:
            device_info['browser'] = 'Other'
        
        # OS detection
        if 'windows' in user_agent_lower:
            device_info['os'] = 'Windows'
        elif 'mac' in user_agent_lower:
            device_info['os'] = 'macOS'
        elif 'linux' in user_agent_lower:
            device_info['os'] = 'Linux'
        elif 'android' in user_agent_lower:
            device_info['os'] = 'Android'
        elif any(term in user_agent_lower for term in ['ios', 'iphone', 'ipad']):
            device_info['os'] = 'iOS'
        else:
            device_info['os'] = 'Other'
    
    return device_info


def get_ip_address(request: Any) -> Optional[str]:
    """
    Get IP address from request headers.
    
    Args:
        request: FastAPI/Starlette request object
        
    Returns:
        IP address string or None
    """
    if not request:
        return None
    
    # Check various headers for IP address
    x_forwarded_for = request.headers.get('x-forwarded-for')
    if x_forwarded_for:
        return x_forwarded_for.split(',')[0].strip()
    
    x_real_ip = request.headers.get('x-real-ip')
    if x_real_ip:
        return x_real_ip
    
    # Try to get from client
    if hasattr(request, 'client') and request.client:
        return request.client.host
    
    return None


async def log_event(
    user_id_or_client_id: str,
    event_type: str,
    category: str,
    action: str,
    metadata: Optional[Dict[str, Any]] = None,
    request: Any = None,
    options: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Centralized event logging function.
    
    Args:
        user_id_or_client_id: User ID (for staff) or Client ID (for clients)
        event_type: Event type (tool_usage, feature_access, etc.)
        category: Category (ESRP, ALE, Chat, etc.)
        action: Action (file_processed, calculation_done, etc.)
        metadata: Additional metadata dictionary
        request: FastAPI/Starlette request object (optional, for device info and role detection)
        options: Additional options dict with keys:
            - clientId: Client ID (if different from user_id_or_client_id)
            - sessionId: Session identifier
            - duration: Event duration in milliseconds
            - status: Event status (success, error, pending)
            - errorMessage: Error message if status is error
    
    Returns:
        Dict containing the created event document, or None if failed
    """
    try:
        if metadata is None:
            metadata = {}
        if options is None:
            options = {}
        
        # Determine if this is a client or user event
        role = None
        if request and hasattr(request, 'state') and hasattr(request.state, 'user_info'):
            role = request.state.user_info.get('role')
        
        is_client = role == 'client'
        
        # Build event data
        event_data = {
            'eventType': event_type,
            'category': category,
            'action': action,
            'metadata': metadata,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        
        # Add user or client ID
        if is_client:
            try:
                event_data['clientId'] = ObjectId(user_id_or_client_id)
            except Exception:
                event_data['clientId'] = user_id_or_client_id
        else:
            try:
                event_data['userId'] = ObjectId(user_id_or_client_id)
            except Exception:
                event_data['userId'] = user_id_or_client_id
        
        # Add request-based information if available
        if request:
            event_data['ipAddress'] = get_ip_address(request)
            event_data['userAgent'] = request.headers.get('user-agent') if hasattr(request, 'headers') else None
            event_data['deviceInfo'] = get_device_info(request)
        
        # Add optional fields from options
        if 'sessionId' in options:
            event_data['sessionId'] = options['sessionId']
        if 'duration' in options:
            event_data['duration'] = options['duration']
        if 'status' in options:
            event_data['status'] = options['status']
        if 'errorMessage' in options:
            event_data['errorMessage'] = options['errorMessage']
        if 'clientId' in options:
            try:
                event_data['clientId'] = ObjectId(options['clientId'])
            except Exception:
                event_data['clientId'] = options['clientId']
        
        # Insert into MongoDB
        db = get_database()
        collection = db['analytics_events']
        result = await collection.insert_one(event_data)
        
        if result.inserted_id:
            log_message('info', f"Analytics event logged: {event_type}/{category}/{action} for {'client' if is_client else 'user'}: {user_id_or_client_id}")
            event_data['_id'] = result.inserted_id
            return event_data
        else:
            log_message('warning', f"Failed to log analytics event: {event_type}/{category}/{action}")
            return None
            
    except Exception as error:
        log_message('error', f"Error logging analytics event: {error}")
        # Don't throw - analytics logging should not break main functionality
        return None


async def create_analytics_indexes():
    """
    Create indexes on the analytics_events collection for better query performance.
    Call this once during application startup.
    """
    try:
        db = get_database()
        collection = db['analytics_events']
        
        # Create indexes matching the Node.js schema
        await collection.create_index([('userId', 1), ('created_at', -1)])
        await collection.create_index([('clientId', 1), ('created_at', -1)])
        await collection.create_index([('eventType', 1), ('created_at', -1)])
        await collection.create_index([('category', 1), ('created_at', -1)])
        await collection.create_index([('created_at', -1)])
        await collection.create_index([('sessionId', 1)])
        await collection.create_index([('userId', 1), ('eventType', 1), ('created_at', -1)])
        await collection.create_index([('clientId', 1), ('eventType', 1), ('created_at', -1)])
        
        log_message('info', 'Created indexes on analytics_events collection')
        
    except Exception as e:
        log_message('warning', f'Failed to create analytics_events indexes (may already exist): {e}')

