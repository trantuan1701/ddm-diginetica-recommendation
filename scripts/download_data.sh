# Note: Please install necessary packages before running this script

# python packages

# direnv: this is what we will use to load env variables. 
# For this one, create a file called `.envrc` and paste the content:
#
# dotenv
#

dvc remote add gdrive gdrive://$GOOGLE_DRIVE_ID/dvc
dvc remote modify gdrive gdrive_acknowledge_abuse true
dvc remote modify --local gdrive gdrive_client_id $GOOGLE_DRIVE_CLIENT_ID
dvc remote modify --local gdrive gdrive_client_secret $GOOGLE_DRIVE_CLIENT_SECRET

dvc pull