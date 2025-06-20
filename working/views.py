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
    

model = None
model_loaded_successfully = False

try:
    # Attempt to load your CatBoost model from 'cat_model.pkl'
    # IMPORTANT: Ensure 'cat_model.pkl' is in the 'predictor' app directory.
    MODEL_PATH = os.path.join(settings.BASE_DIR, 'predictor', 'cat_model.pkl')
    with open(MODEL_PATH, 'rb') as file:
        model = joblib.load(file)
    model_loaded_successfully = True
    print(f"CatBoost model loaded successfully from {MODEL_PATH}.")

except FileNotFoundError:
    print("Warning: cat_model.pkl not found. Prediction will not be possible.")
except Exception as e:
    print(f"Error loading cat_model.pkl: {e}. Prediction will not be possible.")


# --- The 'home' view from your example (kept for structural consistency if you need it) ---
# You can remove this if you only need the prediction page.
# my_attrition_project/working/views.py

# my_attrition_project/working/views.py

import os
import io
import pandas as pd
import joblib
import numpy as np # Make sure numpy is imported for np.where etc.
from django.conf import settings
from django.shortcuts import render, redirect
from django.urls import reverse

# Initialize model globally.
model = None
model_loaded_successfully = False

try:
    # Attempt to load your XGBoost model from 'xgb_model.pkl'
    MODEL_PATH = os.path.join(settings.BASE_DIR, 'working', 'xgb_model.pkl')
    
    print(f"Attempting to load model from: {MODEL_PATH}") # Diagnostic print

    with open(MODEL_PATH, 'rb') as file:
        model = joblib.load(file)
    model_loaded_successfully = True
    print(f"XGBoost model loaded successfully from {MODEL_PATH}.")

except FileNotFoundError:
    print(f"ERROR: xgb_model.pkl not found at {MODEL_PATH}. Prediction will not be possible.")
except Exception as e:
    print(f"ERROR: Failed to load xgb_model.pkl from {MODEL_PATH}: {e}. Prediction will not be possible.")


def home(request):
    print("Welcome to the Attrition Prediction Home Page!")
    return render(request, "index.html")

def predict_attrition(request):
    original_table_html = None
    predicted_table_html = None
    error_message = None
    original_file_name = None

    if request.method == 'GET':
        original_table_html = request.session.get('original_csv_html_attrition', None)
        if 'predicted_csv_html_attrition' in request.session:
            del request.session['predicted_csv_html_attrition']
        if 'original_file_name' in request.session:
            del request.session['original_file_name']

    if request.method == 'POST':
        action = request.POST.get('action')

        if 'original_csv_html_attrition' in request.session:
            del request.session['original_csv_html_attrition']
        if 'predicted_csv_html_attrition' in request.session:
            del request.session['predicted_csv_html_attrition']

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

                    request.session['original_csv_html_attrition'] = df.to_html(
                        index=False,
                        classes='table table-striped table-bordered'
                    )
                    original_table_html = request.session['original_csv_html_attrition']


                    if action == 'predict':
                        if not model_loaded_successfully:
                            error_message = "Critical Error: The actual ML model (xgb_model.pkl) could not be loaded. Prediction cannot be performed without the trained model. Please ensure the model file is correctly placed and valid."
                            predicted_table_html = None
                        else:
                            # --- START: FEATURE ENGINEERING (MUST MATCH TRAINING) ---

                            # 1. Drop irrelevant columns (if they exist in the input CSV)
                            columns_to_drop = ['EmployeeNumber', 'StandardHours', 'Over18']
                            # Remove columns only if they actually exist in the DataFrame
                            df_processed = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

                            # 2. Identify categorical columns for one-hot encoding
                            #    This list MUST be accurate based on your friend's ACC.ipynb
                            categorical_features_for_ohe = [
                                'BusinessTravel', 'Department', 'EducationField', 'Gender',
                                'JobRole', 'MaritalStatus', 'OverTime'
                            ]

                            # 3. Perform One-Hot Encoding
                            #    'drop_first=True' is commonly used. Verify if your friend used it.
                            #    'dtype=int' makes the new columns 0/1 integers.
                            for col in categorical_features_for_ohe:
                                if col not in df_processed.columns:
                                    print(f"Warning: Categorical column '{col}' not found in uploaded CSV. Skipping one-hot encoding for this column.")
                            
                            df_encoded = pd.get_dummies(
                                df_processed,
                                columns=[col for col in categorical_features_for_ohe if col in df_processed.columns],
                                drop_first=True,
                                dtype=int
                            )
                            
                            # --- END: FEATURE ENGINEERING ---


                            # 4. Define the FINAL list of features that were fed into the TRAINED MODEL
                            #    This list MUST contain EXACTLY 30 column names, in the EXACT order,
                            #    as expected by your xgb_model.pkl.
                            #    *** YOU NEED TO GET THIS LIST FROM YOUR FRIEND'S ACC.ipynb ***
                            #    (e.g., from X_train.columns or model.feature_names_in_)
                            #    This is a placeholder list; replace it with the actual 30 feature names.
                            final_model_features = [
                                'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount',
                                'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel',
                                'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
                                'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
                                'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
                                'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
                                'YearsSinceLastPromotion', 'YearsWithCurrManager',
                                # These are example one-hot encoded columns.
                                # Replace with your actual 1-hot encoded columns + numerical ones from friend's model!
                                'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
                                'Department_Research & Development', 'Department_Sales',
                                'EducationField_Life Sciences', 'EducationField_Marketing'
                                # ... you will need 4 more columns here to reach 30 if using the OHE from analysis,
                                # or more depending on original features + drop_first=True/False
                            ]

                            # Check if the number of features matches BEFORE creating features_df
                            if len(final_model_features) != 30:
                                error_message = f"Critical Error: The 'final_model_features' list in views.py is not 30 features long. It has {len(final_model_features)}. It MUST match the 30 features your model expects."
                                predicted_table_html = None
                            elif not all(col in df_encoded.columns for col in final_model_features):
                                missing_cols_for_model = [col for col in final_model_features if col not in df_encoded.columns]
                                error_message = f"Critical Error: After preprocessing, some required model features are missing from the input data. Missing: {', '.join(missing_cols_for_model)}. Please ensure your CSV and preprocessing match the training data."
                                predicted_table_html = None
                            else:
                                # Ensure feature order matches the trained model's expectation
                                features_df = df_encoded[final_model_features].copy()
                                features_array = features_df.values

                                # Perform prediction using the loaded XGBoost model
                                predictions = model.predict(features_array)
                                print("Performing prediction using loaded XGBoost model.")

                                # Add predicted attrition as a new column to the ORIGINAL DataFrame (df)
                                # This ensures the final displayed table includes all original columns plus prediction
                                df['Predicted_Attrition'] = np.where(predictions == 1, 'Yes', 'No')

                                request.session['predicted_csv_html_attrition'] = df.to_html(
                                    index=False,
                                    classes="table table-striped table-bordered"
                                )
                                predicted_table_html = request.session['predicted_csv_html_attrition']

                    else: # If 'view_original' was clicked
                        predicted_table_html = None
                        original_table_html = request.session.get('original_csv_html_attrition', None)

                except pd.errors.EmptyDataError:
                    error_message = "Uploaded CSV file is empty."
                except Exception as e:
                    error_message = f"An error occurred while processing the CSV file or making predictions: {e}"
        else:
            error_message = "No CSV file uploaded. Please select a file."
            if 'original_csv_html_attrition' in request.session:
                del request.session['original_csv_html_attrition']
            if 'predicted_csv_html_attrition' in request.session:
                del request.session['predicted_csv_html_attrition']
            if 'original_file_name' in request.session:
                del request.session['original_file_name']

    original_file_name = request.session.get('original_file_name', None)
    if request.method == 'GET' and not error_message:
         predicted_table_html = request.session.get('predicted_csv_html_attrition', None)
         original_table_html = request.session.get('original_csv_html_attrition', None)


    context = {
        'original_table_html': original_table_html,
        'predicted_table_html': predicted_table_html,
        'error_message': error_message,
        'model_loaded': model_loaded_successfully,
        'original_file_name': original_file_name,
    }
    return render(request, "detail.html", context)