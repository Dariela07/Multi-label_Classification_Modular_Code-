class Config:
    # Input Columns
    TICKET_SUMMARY = 'Ticket Summary'
    INTERACTION_CONTENT = 'Interaction content'

    # Type Columns to test
    TYPE_COLS = ['y2', 'y3', 'y4']
    # CLASS_COL = 'y2'
    CLASS_COL = ['y2', 'y3', 'y4']
    GROUPED = 'y1'
    EPOCHS = 30
    VERBOSE = 1
    TEST_SIZE = 0.3
