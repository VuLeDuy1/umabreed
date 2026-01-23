import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from itertools import product

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173"]}})
Data_dir = './data/'

# --- GLOBAL DATA CONTAINERS ---
name_to_id = {}
char_to_rels = {}
rel_to_val = {}

def load_data():
    global name_to_id, char_to_rels, rel_to_val
    try:
        Character = pd.read_csv(Data_dir + 'Character_Name_ID.csv',
                                names=['ID','Name','is_available'],
                                dtype={'ID': int, 'is_available': bool}).dropna()
        Relations = pd.read_csv(Data_dir + 'Relations.csv',
                                names=['Index', 'Relation_ID', 'Character_ID'],
                                dtype={'Relation_ID': int, 'Character_ID': int})
        Relation_Value = pd.read_csv(Data_dir + 'Relation_Value.csv',
                                    names=['Relation_ID', 'Value'],
                                    dtype={'Relation_ID': int, 'Value': int})

        name_to_id = Character[Character['is_available']].set_index('Name')['ID'].to_dict()      
        char_to_rels = Relations.groupby('Character_ID')['Relation_ID'].apply(set).to_dict()
        rel_to_val = Relation_Value.set_index('Relation_ID')['Value'].to_dict()
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading CSV files: {e}")

# --- CORE LOGIC ---

def get_character_affinity(*character_names) -> int:
    character_ids = [name_to_id.get(name) for name in character_names]
    if None in character_ids or len(set(character_ids)) < len(character_ids):
        return 0
    
    common_rels = char_to_rels.get(character_ids[0], set()).copy()
    for cid in character_ids[1:]:
        common_rels.intersection_update(char_to_rels.get(cid, set()))
    
    result = sum(rel_to_val.get(r, 0) for r in common_rels)
    return result

def calculate_compatibility(*lineage):
    if lineage is None:
        return None
    child, p1, p2, gp1_1, gp1_2, gp2_1, gp2_2 = lineage

    aff_pp = get_character_affinity(p1, p2)
    aff_p1 = get_character_affinity(child, p1)
    ip1 = aff_p1 + get_character_affinity(p1, gp1_1) + get_character_affinity(p1, gp1_2) + aff_pp
    aff_p2 = get_character_affinity(child, p2)
    ip2 = aff_p2 + get_character_affinity((p2, gp2_1)) + get_character_affinity(p2, gp2_2) +aff_pp
    
    igp1_1 = get_character_affinity(child, p1, gp1_1)
    igp1_2 = get_character_affinity(child, p1, gp1_2)
    igp2_1 = get_character_affinity(child, p2, gp2_1)
    igp2_2 = get_character_affinity(child, p2, gp2_2)

    return {
        'P1': ip1,
        'P2': ip2,
        'GP1_1': igp1_1, 'GP1_2': igp1_2,
        'GP2_1': igp2_1, 'GP2_2': igp2_2,
        'Total compatibility': ip1 + ip2 + igp1_1 + igp1_2 + igp2_1 + igp2_2,
        'Displayed affinity': aff_p1 + aff_p2 + aff_pp + igp1_1 + igp1_2 + igp2_1 + igp2_2,
        'lineage': list(lineage)
    }

def find_optimal_lineage(lineage_names, available_names):
    def find_optimal_lineage_for_child(lineage_names, available_names):
        child_name = lineage_names[0]
        fixed_parents = lineage_names[1:3]
        fixed_gps = lineage_names[3:7]

        parent_names = set(available_names + fixed_parents) - {child_name,None}
        gp_names = set(available_names + fixed_gps) - {None}

        p1_names = set([fixed_parents[0]]) if fixed_parents[0] else parent_names
        p2_names = set([fixed_parents[1]]) if fixed_parents[1] else parent_names

        p1p2_names = p1_names | p2_names

        aff_to_child = {n: get_character_affinity(child_name, n) for n in p1p2_names}
        best_halves = {}

        def gp_aff_score(c,p,gp):
            return get_character_affinity(c,p,gp) + get_character_affinity(p,gp)

        for p in p1p2_names:
            gp_scores = []
            for gp in gp_names:
                if gp == p: continue
                score = gp_aff_score(child_name,p,gp)
                gp_scores.append((score, gp))
            
            if len(gp_scores) < 2: continue

            gp_scores.sort(key=lambda x: x[0], reverse=True)
            top_score = [s for s, name in gp_scores[:2]]
            top_gps = [name for s, name in gp_scores[:2]]
            best_halves[p] = (top_score, top_gps)


        def build_halves(child_name, p_names, gp1, gp2, best_halves):
            def apply_fixed_gp(p, fixed_gp, preferred_index, half):
                scores, gps = half
                assert len(scores) == 2 and len(gps) == 2

                if fixed_gp == gps[preferred_index]:
                    return half

                if fixed_gp == gps[1 - preferred_index]:
                    return (scores[::-1], gps[::-1])

                # fixed_gp not present at all
                new_scores = list(scores)
                new_gps = list(gps)

                new_scores[preferred_index] = gp_aff_score(child_name, p, fixed_gp)
                new_gps[preferred_index] = fixed_gp

                return new_scores, new_gps

            halves = {}
            if gp1 and gp2:
                for p in p_names - {gp1, gp2}:
                    halves[p] = (
                        [
                            gp_aff_score(child_name, p, gp1),
                            gp_aff_score(child_name, p, gp2),
                        ],
                        [gp1,gp2]
                    )

            elif gp1:
                for p in p_names - {gp1}:
                    halves[p] = apply_fixed_gp(p,gp1,preferred_index=0,half=best_halves[p])

            elif gp2:
                for p in p_names - {gp2}:
                    halves[p] = apply_fixed_gp(p,gp2,preferred_index=1,half=best_halves[p])

            else:
                return dict(best_halves)

            return halves

        halves1 = build_halves(child_name,p1_names,fixed_gps[0],fixed_gps[1],best_halves)
        halves2 = build_halves(child_name,p2_names,fixed_gps[2],fixed_gps[3],best_halves)

        best_total_score = -1

        best_lineage_result = None

        for p1, p2 in product(p1_names,p2_names):
            if p1 == p2: continue

            half1 = halves1.get(p1,None)
            half2 = halves2.get(p2,None)
            if half1 is None or half2 is None: continue

            scores_p1, gps_p1 = half1
            scores_p2, gps_p2 = half2

            aff_p1_p2 = get_character_affinity(p1, p2)
            
            current_score = aff_to_child[p1] + aff_to_child[p2] + 2 * aff_p1_p2 + sum(scores_p1) + sum(scores_p2)
            
            if current_score > best_total_score:
                best_total_score = current_score
                best_lineage_result = (child_name, p1, p2, gps_p1[0], gps_p1[1], gps_p2[0], gps_p2[1])

        if best_lineage_result is None:
            # print("No possible lineage configuration. Character filter too narrow")
            return None
        return calculate_compatibility(*best_lineage_result)

    if lineage_names is None:
        return None
    if len(lineage_names)!=7:
        return None
    if lineage_names[0]:
        return find_optimal_lineage_for_child(lineage_names, available_names)
    

    fixed_parents = {n for n in lineage_names[1:3] if n is not None}
    best_score = -1
    best_lineage = None
    for c in set(available_names) - fixed_parents:
        partial_lineage = [c] + lineage_names[1:7]
        result = find_optimal_lineage_for_child(partial_lineage, available_names)
        
        if result is None: continue

        if result['Total compatibility'] > best_score:
            best_score = result['Total compatibility']
            best_lineage = result['lineage']
    return calculate_compatibility(*best_lineage)

# --- API ENDPOINTS ---

#---Get Character Names Endpoint ---
@app.route('/characters', methods=['GET'])
def get_characters():
    return jsonify(list(name_to_id.keys()))

# --- Get Affinity Endpoint ---
# input json {'character_name': str}
@app.route('/affinity', methods=['POST'])
def affinity():
    data = request.json
    name = data.get('character_name')
    other_names = name_to_id.keys() - {name}
    affinities = {other: get_character_affinity(name, other) for other in other_names} 
    return jsonify(affinities)
# output shape {name 1: int,
#   name 2: int,
#   ...}

# --- Calculate Lineage Affinity Endpoint ---
# input json {
#     'lineage': [str]  # List of 7 character names: [child, p1, p2, gp1_1, gp1_2, gp2_1, gp2_2]}
@app.route('/lineage_stats', methods=['POST'])
def lineage_stats():
    data = request.json
    lineage = data.get('lineage', [])
    if len(lineage) != 7:
        return jsonify({"error": "exactly 7 character names are required"}), 400
    result = calculate_compatibility(*lineage)
    return jsonify(result)
# output shape {
#     'P1': int,
#     'P2': int,
#     'GP1_1': int,
#     'GP1_2': int,
#     'GP2_1': int,
#     'GP2_2': int,
#     'Total compatibility': int,
#     'Displayed affinity': int,
#     'lineage': [child, p1, p2, gp1_1, gp1_2, gp2_1, gp2_2]}



# --- Optimize Lineage Endpoint ---
# input json {
#     'lineage_names': [str],  # List of 7 character names: [child, p1, p2, gp1_1, gp1_2, gp2_1, gp2_2] (use '' for empty)
#     'available_names': [str]  # Optional}
@app.route('/optimize', methods=['POST'])
def optimize():
    data = request.json
    lineage_names = data.get('lineage_names', [])
    if len(lineage_names) != 7:
        return jsonify({"error": "exactly 7 character names are required"}), 400
    # Default to all names if available_names not provided
    available_names = data.get('available_names', list(name_to_id.keys()))
        
    result = find_optimal_lineage(lineage_names, available_names)

    return jsonify(result)
# output shape {
#     'P1': int,
#     'P2': int,
#     'GP1_1': int,
#     'GP1_2': int,
#     'GP2_1': int,
#     'GP2_2': int,
#     'Total compatibility': int,
#     'Displayed affinity': int,
#     'lineage': [child_name, p1_name, p2_name, gp1_1_name, gp1_2_name, gp2_1_name, gp2_2_name]}


load_data()

if __name__ == '__main__':
    app.run(debug=False, port=5000)