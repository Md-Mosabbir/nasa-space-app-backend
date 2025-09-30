ACTIVITIES = {
    'fishing': {
        'description': "Mild weather, water-friendly activity",
        'DI_preferred': [18, 26]
    },
    'hiking': {
        'description': "Broad comfort for outdoors",
        'DI_preferred': [15, 28]
    },
    'festival': {
        'description': "Social outdoor events",
        'DI_preferred': [20, 30]
    },
    'generic': {
        'description': "Default generic activity",
        'DI_preferred': [16, 28]
    }
}

def list_activities():
    """Return all activities with metadata"""
    return ACTIVITIES
