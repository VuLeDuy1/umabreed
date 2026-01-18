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
    child_name = lineage_names[0]
    fixed_parents = [ p for p in lineage_names[1:3] if p]
    fixed_gps = [ gp for gp in lineage_names[3:] if gp]

    other_names = [n for n in available_names if n != child_name]
    available_parent_names = set(other_names + fixed_parents)
    available_gp_names = set(other_names + fixed_gps)

    if len(other_names) < 2: # Need 2
        return {"error": "Not enough characters available"}

    aff_to_child = {n: get_character_affinity(child_name, n) for n in other_names}
    best_halves = {}

    def gp_aff_score(c,p,gp):
        return get_character_affinity(c,p,gp) + get_character_affinity(p,gp)

    for p in available_parent_names:
        gp_scores = []
        for gp in available_gp_names:
            if gp == p: continue
            score = gp_aff_score(child_name,p,gp)
            gp_scores.append((score, gp))
        
        gp_scores.sort(key=lambda x: x[0], reverse=True)
        top_score = [s for s, name in gp_scores[:2]]
        top_gps = [name for s, name in gp_scores[:2]]
        best_halves[p] = (top_score, top_gps)

    best_total_score = -1
    best_lineage_result = []

    for p1, p2 in product(available_parent_names, repeat=2):
        if p1 == p2: continue
        if lineage_names[1] and p1 != lineage_names[1]: continue
        if lineage_names[2] and p2 != lineage_names[2]: continue

        scores_p1, gps_p1 = best_halves[p1]

        if lineage_names[3] and lineage_names[4]:
            gps_p1 = [lineage_names[3], lineage_names[4]]
            scores_p1 = [gp_aff_score(child_name,p1,lineage_names[3]),
                         gp_aff_score(child_name,p1,lineage_names[4])]
        elif lineage_names[3] and (lineage_names[3] != gps_p1[0]):
            if lineage_names[3] == gps_p1[1]:
                gps_p1[1], gps_p1[0] = gps_p1[0], gps_p1[1]
                scores_p1[1], scores_p1[0] = scores_p1[0], scores_p1[1]
            else:
                gps_p1[1] = lineage_names[3]
                scores_p1[1] = gp_aff_score(child_name,p1,lineage_names[3])
        elif lineage_names[4] and (lineage_names[4] != gps_p1[1]):
            if lineage_names[4] == gps_p1[0]:
                gps_p1[1], gps_p1[0] = gps_p1[0], gps_p1[1]
                scores_p1[1], scores_p1[0] = scores_p1[0], scores_p1[1]
            else:
                gps_p1[1] = lineage_names[4]
                scores_p1[1] = gp_aff_score(child_name,p1,lineage_names[4])

        scores_p2, gps_p2 = best_halves[p2]
        if lineage_names[5] and lineage_names[6]:
            gps_p2 = [lineage_names[5], lineage_names[6]]
            scores_p2 = [gp_aff_score(child_name,p2,lineage_names[5]),
                         gp_aff_score(child_name,p2,lineage_names[6])]
        elif lineage_names[5] and (lineage_names[5] != gps_p2[0]):
            if lineage_names[5] == gps_p2[1]:
                gps_p2[1], gps_p2[0] = gps_p2[0], gps_p2[1]
                scores_p2[1], scores_p2[0] = scores_p2[0], scores_p2[1]
            else:
                gps_p2[1] = lineage_names[5]
                scores_p2[1] = gp_aff_score(child_name,p2,lineage_names[5])
        elif lineage_names[6] and (lineage_names[6] != gps_p2[1]):
            if lineage_names[6] == gps_p2[0]:
                gps_p2[1], gps_p2[0] = gps_p2[0], gps_p2[1]
                scores_p2[1], scores_p2[0] = scores_p2[0], scores_p2[1]
            else:
                gps_p2[1] = lineage_names[6]
                scores_p2[1] = gp_aff_score(child_name,p2,lineage_names[6])

        aff_p1_p2 = get_character_affinity(p1, p2)
        
        current_score = aff_to_child[p1] + aff_to_child[p2] + 2 * aff_p1_p2 + sum(scores_p1) + sum(scores_p2)
        
        if current_score > best_total_score:
            best_total_score = current_score
            best_lineage_result = (child_name, p1, p2, gps_p1[0], gps_p1[1], gps_p2[0], gps_p2[1])

    return calculate_compatibility(*best_lineage_result)

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