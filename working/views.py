# views.py

from django.shortcuts import render
from django.http import HttpResponse
import joblib
import numpy as np
import pandas as pd # Import pandas for CSV handling and data manipulation
import io # For handling in-memory file operations

from django.contrib import admin

from mlmodelplatform import settings
from .models.signup import User_Detail

from django.core.files.storage import FileSystemStorage
import os
from PIL import Image      # For opening images
import imagehash     


# --- Default Model Parameters (PLACEHOLDERS - REPLACE WITH ACTUAL TRAINED VALUES) ---
# These values are illustrative. You MUST replace them with the values obtained
# from training your model in Sales_predction.ipynb.
# If your model.pkl contains these, the code will use those.
# Otherwise, these defaults will be used.


# Example: Assuming 3 features (TV, Radio, Newspaper)
DEFAULT_W_FINAL = np.array([3.9191, 2.7899, -0.0203]) # Example weights
DEFAULT_B_FINAL = 14.0219                            # Example bias
DEFAULT_MEAN = np.array([147.0425, 23.264, 30.554])  # Example mean for TV, Radio, Newspaper
DEFAULT_STD = np.array([85.63933176, 14.80964564, 21.72410606])    # Example standard deviation

# -----------------------------------------------------------------------------------

# Load the trained model or use defaults
w_final = DEFAULT_W_FINAL
b_final = DEFAULT_B_FINAL
mean_features = DEFAULT_MEAN
std_features = DEFAULT_STD
model_loaded_successfully = False

try:
    # Attempt to load model parameters from a saved file
    loaded_model_data = joblib.load('model.pkl')

    if isinstance(loaded_model_data, dict):
        if 'w_final' in loaded_model_data and 'b_final' in loaded_model_data and \
           'mean' in loaded_model_data and 'std' in loaded_model_data:
            w_final = loaded_model_data['w_final']
            b_final = loaded_model_data['b_final']
            mean_features = loaded_model_data['mean']
            std_features = loaded_model_data['std']
            model_loaded_successfully = True
            print("Model parameters loaded from model.pkl.")
        else:
            print("Warning: model.pkl found but missing expected keys (w_final, b_final, mean, std). Using default values.")
    else:
        print("Warning: model.pkl content is not a dictionary. Using default values.")

except FileNotFoundError:
    print("Warning: model.pkl not found. Using default model parameters for sales prediction.")
except Exception as e:
    print(f"Error loading model.pkl: {e}. Using default model parameters for sales prediction.")


def home(request):
    print("hello hii i am karadiya haiderali")
    return render(request, "index.html")

def HOME(request):
    predicted_data_html = None
    error_message = None
    original_file_name = None

    if request.method == 'POST':
        if 'csv_file' in request.FILES:
            uploaded_file = request.FILES['csv_file']
            original_file_name = uploaded_file.name

            # Ensure the file is a CSV
            if not uploaded_file.name.endswith('.csv'):
                error_message = "Please upload a CSV file."
            else:
                try:
                    # Read the CSV file into a pandas DataFrame
                    # Use io.StringIO to read the file content as a string
                    file_content = uploaded_file.read().decode('utf-8')
                    df = pd.read_csv(io.StringIO(file_content))

                    # Drop 'Unnamed: 0' column if it exists, as seen in your notebook
                    if 'Unnamed: 0' in df.columns:
                        df = df.drop(columns=['Unnamed: 0'])

                    # Prepare features for prediction
                    # Ensure the columns 'TV', 'Radio', 'Newspaper' exist
                    required_columns = ['TV', 'Radio', 'Newspaper']
                    if not all(col in df.columns for col in required_columns):
                        error_message = f"Uploaded CSV must contain columns: {', '.join(required_columns)}"
                    else:
                        features_df = df[required_columns].copy()
                        
                        # Convert to numpy array for prediction
                        features_array = features_df.values

                        # Normalize features
                        # Add a small epsilon to std_features if it might contain zeros
                        safe_std_features = np.where(std_features == 0, 1e-8, std_features)
                        features_normalized = (features_array - mean_features) / safe_std_features

                        # Perform prediction
                        predictions = np.dot(features_normalized, w_final) + b_final

                        # Add predicted sales as a new column to the DataFrame
                        df['Sales'] = predictions.round(2) # Round to 2 decimal places

                        # Convert DataFrame to HTML table for display
                        predicted_data_html = df.to_html(classes="table table-striped table-bordered", index=False)

                except ValueError:
                    error_message = "Error processing numeric values in the CSV. Please ensure all advertising spend columns contain numbers."
                except pd.errors.EmptyDataError:
                    error_message = "Uploaded CSV file is empty."
                except Exception as e:
                    error_message = f"An error occurred while processing the file: {e}"
        else:
            error_message = "No file uploaded."

    context = {
        'predicted_data_html': predicted_data_html,
        'error_message': error_message,
        'model_loaded': model_loaded_successfully,
        'original_file_name': original_file_name,
    }
    return render(request, "HOME.html", context)

def DATA(request):
    return render(request, "DATA.html")

def signup(request):
    return render(request, "signup.html")

def compare_licenses_perceptual(uploaded_path, default_path, hash_size=8, threshold=10):
    """
    Compares two images using perceptual hashing.

    Args:
        uploaded_path (str): Path to the user-uploaded license image.
        default_path (str): Path to the backend's default license image.
        hash_size (int): The size of the hash. Larger means more detail, less tolerance for changes.
                         Commonly 8 or 16.
        threshold (int): The maximum Hamming distance allowed for images to be considered a match.
                         Lower means stricter match. Typical values for aHash/dHash are 0-10.

    Returns:
        bool: True if images are considered a match, False otherwise.
    """
    try:
        # Open images
        with Image.open(uploaded_path) as uploaded_img, Image.open(default_path) as default_img:
            # Calculate perceptual hashes (using average_hash is a good general choice)
            uploaded_hash = imagehash.average_hash(uploaded_img, hash_size=hash_size)
            default_hash = imagehash.average_hash(default_img, hash_size=hash_size)

            # Calculate the Hamming distance between the hashes
            hash_distance = uploaded_hash - default_hash # This calculates Hamming distance

            print(f"Uploaded Hash: {uploaded_hash}")
            print(f"Default Hash:  {default_hash}")
            print(f"Hash Distance: {hash_distance}")
            print(f"Threshold:     {threshold}")

            # Compare the distance to the threshold
            if hash_distance <= threshold:
                return True
            else:
                return False

    except FileNotFoundError:
        print("Error: One of the license files not found for hashing.")
        return False
    except Exception as e:
        print(f"An error occurred during perceptual hash comparison: {e}")
        return False

# --- Your existing userdata_view with license logic integrated ---
def userdata_view(request):
    # Initialize authorization status. This will be passed to the template.
    # It's good to have a default state for GET requests or initial page load.
    is_license_authorized = False 
    
    if request.method == 'POST':
        # 1. Retrieve data from the form
        company_name = request.POST.get('companyName')
        company_phone = request.POST.get('companyNo')
        email = request.POST.get('email')
        department = request.POST.get('department')
        username = request.POST.get('username')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirmPassword')
        profile_photo = request.FILES.get('profilePhoto') # For profile photo file upload
        license_image_file = request.FILES.get('licenseFile') # For license image file upload

        # Store in session (as per your context processor needs)
        request.session['profile_name'] = username
        request.session['companyname'] = company_name

        # 2. Basic Validation (add more robust validation as needed)
        if password != confirm_password:
            # messages.error(request, "Passwords do not match!") # type: ignore
            # Pass authorization status even if there's a validation error
            return render(request, 'signup.html', {'is_license_authorized': is_license_authorized})
        
        if User_Detail.objects.filter(gmail=email).exists():
            # messages.error(request, "Email already registered!") # type: ignore
            return render(request, 'signup.html', {'is_license_authorized': is_license_authorized})
        
        if User_Detail.objects.filter(user_name=username).exists():
            # messages.error(request, "Username already taken!") # type: ignore
            return render(request, 'signup.html', {'is_license_authorized': is_license_authorized})
        
        try:
            # --- License Authorization Logic ---
            if license_image_file:
                fs_licenses = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'temp_uploaded_licenses'))
                
                # Ensure the directory for temporary uploads exists
                if not os.path.exists(fs_licenses.location):
                    os.makedirs(fs_licenses.location)

                uploaded_filename = fs_licenses.save(license_image_file.name, license_image_file)
                uploaded_file_path = os.path.join(fs_licenses.location, uploaded_filename)

                default_license_path = os.path.join(settings.MEDIA_ROOT, 'default_licenses', 'default_license.jpg') 
                # ^^^^^ Make sure 'default_license.jpg' is the exact name of your reference file

                if os.path.exists(default_license_path):
                    is_license_authorized = compare_licenses_perceptual(uploaded_file_path, default_license_path)
                    request.session['license_authorized'] = is_license_authorized
                    if is_license_authorized:
                        print("User's license image matched the default.")
                        # messages.success(request, "License verified!")
                    else:
                        print("User's license image did NOT match the default.")
                        # messages.warning(request, "License could not be verified.")
                else:
                    is_license_authorized = False
                    request.session['license_authorized'] = False
                    print(f"Error: Default license image not found at {default_license_path}")
                    # messages.error(request, "Server error: Default license for comparison not found.")
                
                # Clean up the temporarily uploaded license file after comparison
                if os.path.exists(uploaded_file_path):
                    fs_licenses.delete(uploaded_filename)
                    print(f"Deleted temporary uploaded license: {uploaded_filename}")
            else:
                is_license_authorized = False
                request.session['license_authorized'] = False
                print("No license file was uploaded by the user.")
                # messages.info(request, "No license file was provided.")


            # --- Continue with User_Detail object creation and saving ---
            hashed_password = password # In a real app, use Django's make_password: `from django.contrib.auth.hashers import make_password; hashed_password = make_password(password)`
            
            user = User_Detail(
                companyname=company_name,
                companyphone=company_phone,
                gmail=email,
                department=department,
                user_name=username,
                password=hashed_password, 
                image=profile_photo, # Save the uploaded profile image
                # Assuming your User_Detail model has a field to store authorization status
                # You might add a field like 'is_license_verified = models.BooleanField(default=False)'
                # is_license_verified = is_license_authorized 
            )
            user.save()

            # messages.success(request, "Account created successfully!") # type: ignore 
            # If you want to show success message and then redirect
            return render(request,'HOME') # Make sure 'HOME' is a valid URL name in your urls.py

        except Exception as e:
            # messages.error(request, f"An error occurred: {e}") # type: ignore
            # Pass authorization status even if there's an error
            return render(request, 'signup.html', {'is_license_authorized': is_license_authorized})
    
    # If it's a GET request, just render the empty form
    # Pass initial authorization status (false by default)
    return render(request, 'signup.html', {'is_license_authorized': is_license_authorized})


def about_logout(request):
    return render(request, "about_logout.html")

def login(request):
    return render(request, "login.html")

def logout(request):
     
     if 'profile_name' in request.session:
        print(f"Session before deletion: {request.session.items()}")
        del request.session['profile_name']
        del request.session['companyname']
        print("profile_name deleted from session.")
        print(f"Session after deletion: {request.session.items()}")
        request.session.save()  # Explicitly save the session
        return render(request,"HOME.html")
    

#  below is fn for attrition management 

import os
import io
import joblib
import pandas as pd
import numpy as np
from django.conf import settings
from django.shortcuts import render

# --- Global variables to store the loaded models and transformers ---
scaler = None
encoder = None
cat_model = None
models_loaded_successfully = False

try:
    # Define paths to your .pkl files
    # IMPORTANT: Replace 'your_django_app' with the actual name of your Django app
    APP_NAME = 'working' # <--- Make sure this matches your app's directory name
    SCALER_PATH = os.path.join(settings.BASE_DIR, APP_NAME, 'ml_models', 'scaler.pkl')
    ENCODER_PATH = os.path.join(settings.BASE_DIR, APP_NAME, 'ml_models', 'encoder.pkl')
    CAT_MODEL_PATH = os.path.join(settings.BASE_DIR, APP_NAME, 'ml_models', 'cat_model.pkl') # Assuming CatBoost model

    print(f"Attempting to load scaler from: {SCALER_PATH}")
    with open(SCALER_PATH, 'rb') as file:
        scaler = joblib.load(file)
    print(f"Scaler loaded successfully from {SCALER_PATH}.")

    print(f"Attempting to load encoder from: {ENCODER_PATH}")
    with open(ENCODER_PATH, 'rb') as file:
        encoder = joblib.load(file)
    print(f"Encoder loaded successfully from {ENCODER_PATH}.")

    print(f"Attempting to load cat_model from: {CAT_MODEL_PATH}")
    with open(CAT_MODEL_PATH, 'rb') as file:
        cat_model = joblib.load(file)
    print(f"CatBoost model loaded successfully from {CAT_MODEL_PATH}.")

    models_loaded_successfully = True

except FileNotFoundError as e:
    print(f"ERROR: One or more .pkl files not found: {e}. Prediction will not be possible.")
except Exception as e:
    print(f"ERROR: Failed to load .pkl files: {e}. Prediction will not be possible.")


def home(request):
    # This is your initial home page, if different from the prediction page
    print("Welcome to the Attrition Prediction Home Page!")
    return render(request, "index.html")

def predict_attrition(request):
    original_table_html = None
    predicted_table_html = None
    error_message = None
    original_file_name = None # To display the name of the last uploaded file

    # --- Session Management for initial GET or new file upload ---
    # Clear previous prediction results on initial GET request or when
    # a new file is explicitly uploaded (to avoid stale predictions)
    if request.method == 'GET' and 'predicted_csv_html_attrition' in request.session:
        del request.session['predicted_csv_html_attrition']
    
    # If a new file is uploaded, clear any existing original/predicted data in session
    # This handles switching files and ensuring only the latest is processed
    if request.method == 'POST' and 'csv_file' in request.FILES:
        if 'original_csv_html_attrition' in request.session:
            del request.session['original_csv_html_attrition']
        if 'predicted_csv_html_attrition' in request.session:
            del request.session['predicted_csv_html_attrition']
        if 'original_file_name' in request.session:
            del request.session['original_file_name']

    if request.method == 'POST':
        action = request.POST.get('action') # 'view_original' or 'predict'

        if 'csv_file' in request.FILES:
            uploaded_file = request.FILES['csv_file']
            original_file_name = uploaded_file.name
            request.session['original_file_name'] = original_file_name

            if not uploaded_file.name.endswith('.csv'):
                error_message = "Invalid file type. Please upload a CSV file."
            else:
                try:
                    file_content = uploaded_file.read().decode('utf-8')
                    df = pd.read_csv(io.StringIO(file_content))

                    if 'Unnamed: 0' in df.columns:
                        df = df.drop(columns=['Unnamed: 0'])

                    # Store the original dataframe HTML in session
                    request.session['original_csv_html_attrition'] = df.to_html(
                        index=False,
                        classes='table table-striped table-bordered'
                    )
                    original_table_html = request.session['original_csv_html_attrition'] # Set for immediate display

                    if action == 'predict':
                        if not models_loaded_successfully:
                            error_message = "Critical Error: ML models (scaler, encoder, or cat_model) could not be loaded. Prediction cannot be performed."
                        else:
                            # --- START: FEATURE ENGINEERING (MUST MATCH TRAINING) ---

                            # 1. Drop irrelevant columns (if they exist in the input CSV)
                            # Verify these with your friend's original notebook (ACC.ipynb)
                            columns_to_drop = ['EmployeeNumber', 'StandardHours', 'Over18']
                            df_processed = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

                            # 2. Identify numerical and categorical columns expected by your model
                            #    *** YOU NEED TO GET THESE LISTS FROM YOUR FRIEND'S ACC.ipynb ***
                            #    These should be the columns BEFORE OneHotEncoding and Scaling
                            numerical_features = [
                                'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount',
                                'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel',
                                'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
                                'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
                                'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
                                'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
                                'YearsSinceLastPromotion', 'YearsWithCurrManager'
                            ]
                            categorical_features = [
                                'BusinessTravel', 'Department', 'EducationField', 'Gender',
                                'JobRole', 'MaritalStatus', 'OverTime'
                            ]

                            # Ensure all expected raw columns are present in the uploaded CSV
                            missing_expected_raw_cols = [col for col in numerical_features + categorical_features if col not in df_processed.columns]
                            if missing_expected_raw_cols:
                                error_message = f"Missing required columns in CSV: {', '.join(missing_expected_raw_cols)}. Please check your input file."
                                # Skip further processing if essential columns are missing
                                raise ValueError("Missing required raw columns for prediction.")

                            # Separate numerical and categorical data
                            df_numerical = df_processed[numerical_features]
                            df_categorical = df_processed[categorical_features]

                            # Apply OneHotEncoder
                            try:
                                encoded_features = encoder.transform(df_categorical)
                                # For newer sklearn (0.23+), use get_feature_names_out()
                                encoded_feature_names = encoder.get_feature_names_out(categorical_features)
                                df_encoded = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df_processed.index)
                            except Exception as e:
                                error_message = f"Error during One-Hot Encoding: {e}. Ensure the categorical columns in your CSV match the encoder's training data."
                                raise ValueError(error_message) # Propagate to outer try-except

                            # Apply StandardScaler to numerical features
                            try:
                                scaled_numerical_features = scaler.transform(df_numerical)
                                df_scaled_numerical = pd.DataFrame(scaled_numerical_features, columns=numerical_features, index=df_processed.index)
                            except Exception as e:
                                error_message = f"Error during Standardization: {e}. Ensure numerical columns in your CSV are appropriate for scaling."
                                raise ValueError(error_message) # Propagate to outer try-except

                            # Concatenate processed numerical and encoded categorical features
                            # The order of concatenation here must match the order during training
                            # A common approach is: numerical_features + encoded_features
                            final_features_df = pd.concat([df_scaled_numerical, df_encoded], axis=1)

                            # --- VERY IMPORTANT: ALIGN COLUMNS TO TRAINING ORDER ---
                            # Get the exact feature names and their order from your trained model (cat_model.pkl)
                            # This is usually model.feature_names_in_ if it's a scikit-learn compatible model
                            # Or from X_train.columns if you saved that list during training.
                            # For CatBoost, it often handles feature names internally, but explicit alignment is safest.
                            
                            # Placeholder: You MUST replace this with the ACTUAL list of columns
                            # used for training your cat_model.pkl, IN THAT EXACT ORDER.
                            # Example (replace with actual feature names from your model, usually 40+ features)
                            model_expected_features = [
                                'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount', 'EnvironmentSatisfaction',
                                'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
                                'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
                                'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
                                'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
                                # Encoded features - THESE MUST BE EXACTLY AS GENERATED BY YOUR ENCODER DURING TRAINING
                                'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
                                'Department_Research & Development', 'Department_Sales',
                                'EducationField_Life Sciences', 'EducationField_Marketing', 'EducationField_Medical',
                                'EducationField_Other', 'EducationField_Technical Degree', 'Gender_Male',
                                'JobRole_Human Resources', 'JobRole_Laboratory Technician', 'JobRole_Manager',
                                'JobRole_Manufacturing Director', 'JobRole_Research Director', 'JobRole_Research Scientist',
                                'JobRole_Sales Executive', 'JobRole_Sales Representative',
                                'MaritalStatus_Married', 'MaritalStatus_Single',
                                'OverTime_Yes'
                            ]
                            
                            # Ensure all expected features are present in the `final_features_df`
                            missing_features_for_model = [f for f in model_expected_features if f not in final_features_df.columns]
                            if missing_features_for_model:
                                error_message = f"Error: After preprocessing, expected features for the model are missing: {', '.join(missing_features_for_model)}. This typically means the input CSV schema or preprocessing steps differ from training."
                                raise ValueError(error_message) # Propagate to outer try-except
                            
                            # Reorder columns to match the trained model's expectation
                            features_for_prediction = final_features_df[model_expected_features]
                            
                            # --- END: FEATURE ENGINEERING ---

                            # Perform prediction using the loaded CatBoost model
                            predictions = cat_model.predict(features_for_prediction)
                            # If your model returns probabilities, and you want 0/1:
                            # predictions = (cat_model.predict_proba(features_for_prediction)[:, 1] > 0.5).astype(int)
                            print("Performing prediction using loaded CatBoost model.")

                            # Add predicted attrition as a new column to the ORIGINAL DataFrame (df)
                            df['Predicted_Attrition'] = np.where(predictions == 1, 'Yes', 'No')

                            request.session['predicted_csv_html_attrition'] = df.to_html(
                                index=False,
                                classes="table table-striped table-bordered"
                            )
                            predicted_table_html = request.session['predicted_csv_html_attrition']

                    # If 'view_original' was clicked, or if prediction failed,
                    # just ensure original_table_html is set from session
                    # (it's already set from the uploaded file section above)
                    elif action == 'view_original':
                        predicted_table_html = None # Explicitly clear any stale prediction display
                        original_table_html = request.session.get('original_csv_html_attrition', None)

                except pd.errors.EmptyDataError:
                    error_message = "Uploaded CSV file is empty."
                except ValueError as ve: # Catch custom ValueErrors from missing columns/features/preprocessing
                    error_message = str(ve) # Display the specific error message
                except Exception as e:
                    error_message = f"An unexpected error occurred while processing the CSV file or making predictions: {e}"
        else:
            error_message = "No CSV file uploaded. Please select a file."
            # Clear all session data if no file is uploaded on a POST request
            if 'original_csv_html_attrition' in request.session:
                del request.session['original_csv_html_attrition']
            if 'predicted_csv_html_attrition' in request.session:
                del request.session['predicted_csv_html_attrition']
            if 'original_file_name' in request.session:
                del request.session['original_file_name']

    # Retrieve data from session for display on GET requests or after POST processing
    if not error_message: # Only retrieve if no new error occurred in current POST
        original_table_html = request.session.get('original_csv_html_attrition', None)
        predicted_table_html = request.session.get('predicted_csv_html_attrition', None)
        original_file_name = request.session.get('original_file_name', None)


    context = {
        'original_table_html': original_table_html,
        'predicted_table_html': predicted_table_html,
        'error_message': error_message,
        'model_loaded': models_loaded_successfully,
        'original_file_name': original_file_name,
    }
    return render(request, "detail.html", context) # Your attrition template is detail.html