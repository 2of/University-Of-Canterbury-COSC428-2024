'''
COSC 428


simple state machine; no errors raised on invalid transitions
transition only on valid transitions
defined in transitions below as key value pairs of transition names and new states
'''

transitions = {
    'idle': {
        'point': 'track_location_no_click',
        'pinch_w_click': 'track_location_click',
        'pinch_no_click': 'track_location_no_click',
        '2-finger': 'pan',
        'spread':'spread'
    },
    'click': {
        'point': 'track_location_no_click',
        'idle': 'idle',
        'spread': 'spread',
        '2-finger': 'pan',
        'pinch_no_click': 'track_location_no_click'
    },
    'track_location_no_click': {
        'idle': 'idle',
        'spread': 'spread',
        '2-finger': 'pan',
        'pinch_w_click': 'track_location_click',
        'pinch_no_click': 'pinch_zoom'
    },
    'track_location_click': {
        'idle': 'idle',
        'spread': 'spread',
        '2-finger': 'pan',
        'pinch_no_click': 'pinch_zoom'
    },
    'pinch_zoom': {
        'point': 'track_location_no_click',
        'idle': 'idle',
        'pinch_w_click': 'track_location_click',
        '2-finger': 'pan',
        'spread': 'spread'
    },
    'spread': {
        'point': 'track_location_no_click',
        'pinch_w_click': 'track_location_click',
        '2-finger': 'pan',
        'idle': 'idle'
    },
    'pan': {
        'point': 'track_location_no_click',
        'pinch_w_click': 'track_location_click',
        '2-finger': 'pan',
        'spread': 'spread',
        'idle': 'idle'
    }
}

class StateMachine:
    def __init__(self, transitions=transitions, initial="idle"):
        self.transitions = transitions
        self.state = initial

    def transition(self, event):
        # print("TRANSITION TO", event)
        if self.state in self.transitions and event in self.transitions[self.state]:
            self.state = self.transitions[self.state][event]
        # else:
        #     # print(self.transitions)
        #     print(f"Invalid event {event} for state {self.state}")

    def print_transitions(self):
        print(self.transitions)

    def current_state(self):
        return self.state

    
