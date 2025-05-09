import json
from models import Profile, neo4j_config, embeddings
from tqdm import tqdm
from datetime import datetime


def calculate_age(birthdate_str):
    birthdate = datetime.strptime(birthdate_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    today = datetime.today()
    age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
    return age

def read_profiles_from_json(file_path="mektoube_production.utilisateur.json"):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            profiles = []
            for profile in data:
                age = -1
                if "date_naissance" in profile:
                    age = calculate_age(profile["date_naissance"])
                sex = "U"
                if "id_genre" in profile:
                    sex = "F" if profile["id_genre"] == 2 else "M"
                pays = None
                if "pays" in profile:
                    pays = profile["pays"]
                accroche = None
                if "accroche" in profile:
                    accroche = profile["accroche"]
                profile = Profile(
                    user_id=profile["id_utilisateur"],
                    age=age,
                    ville=profile["ville"],
                    region=profile["region"],
                    pays=pays,
                    bio=accroche,
                    sex=sex
                )
                profiles.append(profile)
            return profiles
    except FileNotFoundError:
        print(f"Error: {file_path} not found")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return []
    except KeyError as e:
        print(f"Error: Missing required field {e} in profile data")
        return []


def import_profiles(profiles: list[Profile]):
    if not profiles:
        print("No profiles to import")
        return

    # Create constraints and indexes
    neo4j_config.create_constraints()
    neo4j_config.create_vector_index()

    with neo4j_config.driver.session() as session:
        for profile in tqdm(profiles, desc="Importing profiles"):
            # Generate embedding for bio
            bio_embedding = None
            if profile.bio:
                bio_embedding = embeddings.embed_query(profile.bio)
            
            # Check if profile already exists
            result = session.run(
                "MATCH (p:Profile {user_id: $user_id}) RETURN p",
                user_id=profile.user_id
            )
            if result.single():
                continue

            # Create profile node with embedding
            session.run(
                """
                CREATE (p:Profile {
                    user_id: $user_id,
                    age: $age,
                    bio: $bio,
                    sex: $sex,
                    bio_embedding: $bio_embedding,
                    ville: $ville,
                    region: $region,
                    pays: $pays
                })
            """,
                user_id=profile.user_id,
                age=profile.age,
                bio=profile.bio,
                sex=profile.sex,
                bio_embedding=bio_embedding,
                ville=profile.ville,
                region=profile.region,
                pays=profile.pays,
            )


if __name__ == "__main__":
    print("Reading profiles from user.json...")
    profiles = read_profiles_from_json()

    if profiles:
        print(f"Found {len(profiles)} profiles to import")
        print("Importing profiles to Neo4j...")
        import_profiles(profiles)
        print("Done!")
    else:
        print("No profiles were imported")

    neo4j_config.close()
