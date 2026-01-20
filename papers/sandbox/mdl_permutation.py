import numpy as np

# 1. THE STATIC SUPERSPACE
# We need a larger pool because as our motif grows, it becomes harder to find matches
POOL_SIZE = 50000 
BIT_LENGTH = 128
superspace = np.random.randint(0, 2, (POOL_SIZE, BIT_LENGTH))

# 2. THE GROWING OBSERVER (The "Life Path")
# We start with a simple "DNA" seed
seed = [1, 0, 1, 0] 

def get_observer_at_age(age):
    """
    Simulates growth: At each 'age', the observer adds a bit 
    representing a new memory or physical structure.
    """
    pattern = list(seed)
    for i in range(age):
        # Let's say we accumulate a consistent pattern of '1's as we grow
        pattern.append(1 if i % 2 == 0 else 0)
    return np.array(pattern)

# 3. THE LIFE-CYCLE WALK
current_state = None
# Find a starting universe that contains the 'infant' observer
infant_motif = get_observer_at_age(0)
for s in superspace:
    if any(np.array_equal(s[i:i+len(infant_motif)], infant_motif) for i in range(BIT_LENGTH - len(infant_motif))):
        current_state = s
        break

print(f"Birth: Observer Seed '{infant_motif}' found in static Superspace.\n")

for age in range(1, 10):
    motif = get_observer_at_age(age)
    
    # FILTER: Find all configurations in the static pool that contain the CURRENT 'Grown' version of us
    valid_candidates = []
    for s in superspace:
        # Check if the grown pattern exists in this static configuration
        if any(np.array_equal(s[i:i+len(motif)], motif) for i in range(BIT_LENGTH - len(motif))):
            valid_candidates.append(s)
    
    if not valid_candidates:
        print(f"Age {age}: DEATH. The superspace no longer supports a pattern this complex.")
        break
        
    # MDL SELECTION: Move to the candidate with the smallest Hamming Distance
    distances = [np.sum(current_state != cand) for cand in valid_candidates]
    current_state = valid_candidates[np.argmin(distances)]
    
    print(f"Age {age}: Pattern '{''.join(map(str, motif))}' | Complexity Increasing...")
    print(f"       -> Perceived 'Universe' State: {current_state[:32]}...")