import streamlit as st
import pandas as pd
from streamlit_extras.metric_cards import style_metric_cards
from database_connector import connect_to_mongodb, retrieve_data_from_collection, close_mongodb_connection
from streamlit_option_menu import option_menu
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from PIL import Image
import plotly.express as px
import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
from numerize.numerize import numerize
import time
from streamlit_extras.metric_cards import style_metric_cards
st.set_option('deprecation.showPyplotGlobalUse', False)
import plotly.graph_objs as go
from database_connector import connect_to_mongodb, retrieve_data_from_collection, close_mongodb_connection
from collections import Counter
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
from numerize.numerize import numerize
import time
from streamlit_extras.metric_cards import style_metric_cards
st.set_option('deprecation.showPyplotGlobalUse', False)
import plotly.graph_objs as go
from database_connector import connect_to_mongodb, retrieve_data_from_collection, close_mongodb_connection
from collections import Counter
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from database_connector import connect_to_mongodb, retrieve_data_from_collection
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from database_connector import connect_to_mongodb, retrieve_data_from_collection, close_mongodb_connection
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

st.set_option('deprecation.showPyplotGlobalUse', False)


st.set_page_config(page_title="Dashboard", page_icon="🌍", layout="wide")

# Database Connectivity
client, db = connect_to_mongodb()

# Retrieve data from MongoDB for each collection
collections = ['CDR', 'USER ACTION', 'TRANSACTION', 'ONBOARD', 'USER LOG', 'did_details']
dataframes = {}
    


# this function performs basic descriptive analytics like Mean, Mode, Sum  etc
def Home():
    # # Centered header using HTML
    # st.markdown("<h1 style='text-align: center;'>MISSCALLPAY AUTOMATION TOOL</h1>", unsafe_allow_html=True)
    # # Load Style css
    # with open('style.css') as f:
    #     st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    # with st.sidebar.container():
    #     image = Image.open(r"C:\Users\miv-003\OneDrive\Automation Dashboard\data\logo.png")
    #     st.image(image, use_column_width=True)
    # Sidebar
    st.sidebar.subheader("Filter Data by Date Range")
    dataset_date = pd.to_datetime("2023-02-27 01:26:38")
    date_range_start = st.sidebar.date_input("Start Date", dataset_date)
    date_range_end = st.sidebar.date_input("End Date", pd.to_datetime("today"))

    # Convert date_range_start and date_range_end to datetime objects
    date_range_start = pd.to_datetime(date_range_start).normalize()
    date_range_end = pd.to_datetime(date_range_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    # Retrieve data from MongoDB for each collection
    # Retrieve data from MongoDB for each collection
    for collection_name in collections:
        dataframes[collection_name] = retrieve_data_from_collection(db, collection_name)
        

        # Filter data based on date range
        date_column = None
        if collection_name == 'CDR':
            date_column = 'start_time'
        elif collection_name == 'ONBOARD':
            date_column = 'call_date'
        elif collection_name == 'USER ACTION':
            date_column = 'datetime'
        elif collection_name == 'USER LOG':
            date_column = 'call_date'
        elif collection_name == 'TRANSACTION':
            date_column = 'initiated_at'

        if date_column:
            dataframes[collection_name][date_column] = pd.to_datetime(dataframes[collection_name][date_column])
            filtered_data = dataframes[collection_name].loc[
                (dataframes[collection_name][date_column] >= date_range_start) &
                (dataframes[collection_name][date_column] <= date_range_end)
            ]
            dataframes[collection_name] = filtered_data
        
    # st.sidebar.image("C:\Users\miv-003\OneDrive\Automation Dashboard\data\logo1.png", use_column_width=True)

    # menu bar

    # Calculate required metrics and statistics
    # ... (calculate necessary metrics and statistics)
    # print(dataframes.keys())

    # ________________________________________________________SIDEBAR.MULTISELECT_________________________________________________________________
    txn_data = dataframes['TRANSACTION']
    did_data = dataframes['did_details']
    cdr_data = dataframes['CDR']

    txn_data = pd.merge(txn_data, did_data[['did', 'page']], left_on='did', right_on='did', how='left')
    txn_data.rename(columns={'page': 'did_name'}, inplace=True)
    # Create a new column 'PSP_type' based on 'did_name' values
    txn_data['PSP_type'] = 'IDFC'  # Default value for all rows
    txn_data.loc[txn_data['did_name'] == 'BOI Live', 'PSP_type'] = 'BOI'
    # Create a new column 'payment_method' based on 'txn_request' values
    txn_data['payment_method'] = 'IVR'  # Default value for all rows
    txn_data.loc[txn_data['txn_request'].isin(['QR', 'UPI APP']), 'payment_method'] = 'QR Code'

    # Create a new column 'Page_Through' based on 'o2o_session_id'
    txn_data['Page_Through'] = 'Direct IVR'
    txn_data.loc[txn_data['o2o_session_id'].notnull(), 'Page_Through'] = 'Landing Page'

    cdr_data = pd.merge(cdr_data, did_data[['did', 'page']], left_on='did', right_on='did', how='left')
    cdr_data.rename(columns={'page': 'did_name'}, inplace=True)
    # Create a new column 'PSP_type' based on 'did_name' values
    cdr_data['PSP_type'] = 'IDFC'  # Default value for all rows
    cdr_data.loc[cdr_data['did_name'] == 'BOI Live', 'PSP_type'] = 'BOI'


    # switcher
    transaction_type = st.sidebar.multiselect(
        "SELECT TRANSACTION TYPE",
        options=txn_data["txn_type"].unique(),
        default=txn_data["txn_type"].unique(),
    )
    psp_type = st.sidebar.multiselect(
        "SELECT PSP TYPE",
        options=txn_data["PSP_type"].unique(),
        default=txn_data["PSP_type"].unique(),
    )

    df_selection_txn = txn_data.query(
        "txn_type == @transaction_type & PSP_type == @psp_type"
    )

    df_selection_cdr = cdr_data.query(
        "PSP_type == @psp_type"
    )

    # Ensure default values are in options
    default_columns = ["id", "bank_txn_id", "txn_amount", "payee_name", "payer_name", "completed_at", "payer_mobile",
                        "bank_txn_result", "txn_type"]
    available_columns_txn = txn_data.columns.tolist()
    valid_default_columns_txn = [col for col in default_columns if col in available_columns_txn]

    with st.expander("VIEW EXCEL DATASET - TRANSACTION"):
        showData_txn = st.multiselect('Filter: ', available_columns_txn, default=valid_default_columns_txn)
        st.dataframe(df_selection_txn[showData_txn], use_container_width=True)

    # Check if the DataFrame is not empty before accessing columns
    if not txn_data.empty:
        # Compute top analytics
        Users_on_boarded = len(dataframes['ONBOARD'])
        total_txn = txn_data['ID'].count()
        # Successful trxns total #(IVR, QR Code, Biller)
        total_success_txn = txn_data[txn_data['bank_txn_result'].isin(['Approved', 'Success'])]['txn_amount'].count()
        # Failed Total Count
        failed_total_count = txn_data[
            ~txn_data['bank_txn_result'].isin(['Approved', 'Success', 'EXPIRED', 'PENDING', ''])]['ID'].count()
        inbound_count = dataframes['CDR'][(dataframes['CDR']['call_direction'].isin(['in']))]['uid'].count()

                # Total Outbound Call
        outbound_count = dataframes['CDR'][(dataframes['CDR']['call_direction'].isin(['out']))]['uid'].count()

        total11, total1, total2, total3, total4, total5 = st.columns(6, gap='small')
        with total11:
            st.info('Onboarded Users', icon="💰")
            st.metric(label="Count Onboard", value=f"{Users_on_boarded:,.0f}")
        with total1:
            st.info('Total Transaction ', icon="💰")
            st.metric(label="Count Txn", value=f"{total_txn:,.0f}")

        with total2:
            st.info('Scuccess Transaction', icon="💰")
            st.metric(label="Count Txn", value=f"{total_success_txn:,.0f}")

        with total3:
            st.info('Failed Transaction', icon="💰")
            st.metric(label="Count Txn", value=f"{failed_total_count:,.0f}")

        with total4:
            st.info('Incoming Calls  ', icon="📞")
            st.metric(label="Count Calls", value=f"{inbound_count:,.0f}")

        with total5:
            st.info('Outbounding Calls  ', icon="📞")
            st.metric(label="Count Calls", value=f"{outbound_count:,.0f}")
        style_metric_cards(background_color="#FFFFFF", border_left_color="#686664", border_color="#000000",
                           box_shadow="#F71938")

        # variable distribution Histogram
        # variable distribution Histogram

        # Plot histograms
        with st.expander("DISTRIBUTIONS BY Successful Transaction Time Period"):
            successful_txn_data = txn_data[txn_data['bank_txn_result'].isin(['Approved', 'Success'])]

            # Convert 'initiated_at' to datetime if it's not already
            successful_txn_data['initiated_at'] = pd.to_datetime(successful_txn_data['initiated_at'])

            # Extract date, month, and year from 'initiated_at'
            successful_txn_data['date'] = successful_txn_data['initiated_at'].dt.date
            successful_txn_data['month'] = successful_txn_data['initiated_at'].dt.to_period('M')
            successful_txn_data['year'] = successful_txn_data['initiated_at'].dt.year

            # Extract hour information
            successful_txn_data['hour'] = successful_txn_data['initiated_at'].dt.hour

            # Plot histograms
            st.subheader("Successful Transaction Time")
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.hist(successful_txn_data['hour'], color='#898784', bins=24, edgecolor='black', linewidth=1.2)
            ax.set_title("Successful Transaction Time")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Count")
            ax.grid(axis='y', alpha=0.75)
            st.pyplot(fig)

            st.subheader("Successful Transaction Day")
            fig, ax = plt.subplots(figsize=(16, 8))
            ax.hist(successful_txn_data['date'], color='#898784', bins=31, edgecolor='black', linewidth=1.2)
            ax.set_title("Successful Transaction Day")
            ax.set_xlabel("Date")
            ax.set_ylabel("Count")
            ax.grid(axis='y', alpha=0.75)
            st.pyplot(fig)

            st.subheader("Successful Transaction Month")
            fig, ax = plt.subplots(figsize=(16, 8))
            successful_txn_data['month'].value_counts().sort_index().plot(kind='bar', color='#898784', ax=ax)
            ax.set_title("Successful Transaction Month")
            ax.set_xlabel("Month")
            ax.set_ylabel("Count")
            ax.grid(axis='y', alpha=0.75)
            st.pyplot(fig)

            st.subheader("Successful Transaction Year")
            fig, ax = plt.subplots(figsize=(16, 8))
            successful_txn_data['year'].value_counts().sort_index().plot(kind='bar', color='#898784', ax=ax)
            ax.set_title("Successful Transaction Year")
            ax.set_xlabel("Year")
            ax.set_ylabel("Count")
            ax.grid(axis='y', alpha=0.75)
            st.pyplot(fig)

    else:
        st.warning("Transaction data is empty. Please check your data.")


    # Filter successful and failed transactions
    successful_txn_data = txn_data[txn_data['bank_txn_result'].isin(['Approved', 'Success'])]
    failed_txn_data = txn_data[~txn_data['bank_txn_result'].isin(['Approved', 'Success', 'EXPIRED', 'PENDING', ''])]

    # Calculate total amount for successful and failed transactions
    total_successful_amount = successful_txn_data['txn_amount'].sum()
    total_failed_amount = failed_txn_data['txn_amount'].sum()

    # Create a Plotly bar chart
    fig1 = px.bar(
        x=['Successful Transactions', 'Failed Transactions'],
        y=[total_successful_amount, total_failed_amount],
        labels={'y': 'Total Amount (INR)'},
        color=['Successful Transactions', 'Failed Transactions'],
        title='Total Amount for Successful and Failed Transactions',
    )
    fig1.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="black"),
        yaxis=dict(showgrid=True, gridcolor='#cecdcd'),  # Show y-axis grid and set its color
        paper_bgcolor='rgba(0, 0, 0, 0)',  # Set paper background color to transparent
        xaxis=dict(showgrid=True, gridcolor='#cecdcd'),  # Show x-axis grid and set its color
    )
   

    # Filter successful transactions
    successful_txn = txn_data[txn_data['bank_txn_result'].isin(['Approved', 'Success'])]

    # Group by payment method and get the count of successful transactions
    payment_method_counts = successful_txn['payment_method'].value_counts()

    # Create a Plotly pie chart
    fig2 = px.pie(
        payment_method_counts,
        names=payment_method_counts.index,
        values=payment_method_counts.values,
        title='Successful Transactions by Payment Method',
        labels={'names': 'Payment Method', 'values': 'Count'},
        hole=0.3,  # Add a hole in the center for better aesthetics
    )

    

    # Filter successful transactions
    successful_txn = txn_data[txn_data['bank_txn_result'].isin(['Approved', 'Success'])]

    # Create a Pie chart
    fig3 = px.pie(
        successful_txn,
        names='Page_Through',
        title='Successful Transaction Count by Landing Page And IVR',
        hole=0.3,
    )

    

    # Filter successful transactions
    successful_txn = txn_data[txn_data['bank_txn_result'].isin(['Approved', 'Success'])]

    # Get the top 10 did_name based on transaction count
    top_10_did = successful_txn['did_name'].value_counts().nlargest(10)
    # Calculate count and percentage for each did_name
    top_10_data = pd.DataFrame({'did_name': top_10_did.index, 'count': top_10_did.values})
    top_10_data['percentage'] = ((top_10_data['count'] / successful_txn.shape[0]) * 100).round(2)

    # Format percentage as strings with % sign
    top_10_data['percentage'] = top_10_data['percentage'].apply(lambda x: f"{x:.2f}%")

    # Create a Plotly bar chart
    fig4 = px.bar(
        top_10_data,
        x='did_name',
        y='count',
        color='did_name',
        text='percentage',
        labels={'count': 'Count', 'percentage': 'Percentage (%)'},
        title='Top 10 Transactions by Merchant (Successful Transactions)',
        width=1400,
        height=700
    )

  

    # Calculate counts for each transaction status
    success = txn_data[txn_data['bank_txn_result'].isin(['Approved', 'Success'])]
    failed = txn_data[~txn_data['bank_txn_result'].isin(['Approved', 'Success', 'EXPIRED', 'PENDING', ''])]
    expired = txn_data[txn_data['bank_txn_result'].isin(['EXPIRED'])]
    pending = txn_data[txn_data['bank_txn_result'].isin(['PENDING'])]

    # Create a Pie chart using Plotly
    fig5 = px.pie(
        names=['Success', 'Failed', 'Expired', 'Pending'],
        values=[len(success), len(failed), len(expired), len(pending)],
        hole=0.3,  # Set the size of the hole in the middle of the pie
        title='Transaction Status Distribution',
        color=['Success', 'Failed', 'Expired', 'Pending'],  # Color pie chart slices
    )

    # Add text annotations for count and percentage on the right side
    annotations = [
        dict(
            text=f"{len(success):,.0f} ({(len(success)/len(txn_data))*100:.2f}%)",
            x=0.85, y=0.75,
            font=dict(color='white', size=12),
            showarrow=False,
        ),
        dict(
            text=f"{len(failed):,.0f} ({(len(failed)/len(txn_data))*100:.2f}%)",
            x=0.85, y=0.5,
            font=dict(color='white', size=12),
            showarrow=False,
        ),
        dict(
            text=f"{len(expired):,.0f} ({(len(expired)/len(txn_data))*100:.2f}%)",
            x=0.85, y=0.25,
            font=dict(color='white', size=12),
            showarrow=False,
        ),
        dict(
            text=f"{len(pending):,.0f} ({(len(pending)/len(txn_data))*100:.2f}%)",
            x=0.85, y=0.0,
            font=dict(color='white', size=12),
            showarrow=False,
        ),
    ]

    # fig5.update_layout(annotations=annotations)

    # Display the chart
    # Arrange charts side by side
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig1)
    col2.plotly_chart(fig5)
    col3, col4 = st.columns(2)
    col3.plotly_chart(fig2)
    col4.plotly_chart(fig3)
    # Display the chart
    st.plotly_chart(fig4)
    

def Data_show():
    # Sidebar
    st.sidebar.subheader("Filter Data by Date Range")

    # Retrieve data from MongoDB for each collection
    for collection_name in collections:
        dataframes[collection_name] = retrieve_data_from_collection(db, collection_name)

    # Check if 'did' column exists in did_details DataFrame
    if 'did' in dataframes.get('did_details', {}).columns:
        # Merge 'did' DataFrame to create 'did_name' column in each DataFrame
        for collection_name, dataframe in dataframes.items():
            if collection_name != 'did_details' and 'did' in dataframe.columns:
                did_data = dataframes['did_details']
                dataframes[collection_name] = pd.merge(dataframe, did_data[['did', 'page']], left_on='did', right_on='did', how='left')
                dataframes[collection_name].rename(columns={'page': 'did_name'}, inplace=True)

    # Single date range filter
    dataset_date = pd.to_datetime("2023-02-27 01:26:38")
    date_range_start = st.sidebar.date_input("Start Date", dataset_date)
    date_range_end = st.sidebar.date_input("End Date", datetime.now())

    # Convert date_range_start and date_range_end to datetime objects
    date_range_start = datetime.combine(date_range_start, datetime.min.time())
    date_range_end = datetime.combine(date_range_end, datetime.max.time())

    # Upload data to MongoDB button and display DataFrame for each collection
    for collection_name in ['CDR', 'USER ACTION', 'TRANSACTION', 'ONBOARD', 'USER LOG']:
        st.subheader(f'Upload Data for {collection_name.capitalize()}')
        uploaded_file = st.file_uploader(f"Upload {collection_name.capitalize()} Data (Max File Size: 200MB)",
                                        type="csv", key=f"{collection_name}_uploader")

        if uploaded_file is not None:
            # Read the uploaded CSV file into a DataFrame
            new_data = pd.read_csv(uploaded_file)

            # Upload data to MongoDB
            collection = db[collection_name]
            collection.insert_many(new_data.to_dict(orient='records'))

            st.success(f"Data uploaded to {collection_name} collection in MongoDB.")

        # Display data for the current collection and create plots
        st.subheader(f'{collection_name.capitalize()} Data')
        dataframe = dataframes.get(collection_name, pd.DataFrame())

        if collection_name != 'did_details':
            date_column = None

            if collection_name == 'CDR':
                date_column = 'start_time'
            elif collection_name == 'ONBOARD':
                date_column = 'call_date'
            elif collection_name == 'USER ACTION':
                date_column = 'datetime'
            elif collection_name == 'USER LOG':
                date_column = 'call_date'
            elif collection_name == 'TRANSACTION':
                date_column = 'initiated_at'

            if date_column:
                # Convert the column to datetime objects
                dataframe[date_column] = pd.to_datetime(dataframe[date_column])

                # Remove rows with disposition 'ANSWERED' for CDR
                if collection_name == 'CDR':
                    dataframe = dataframe[dataframe['disposition'] != 'ANSWERED']

                # Apply date range filter
                filtered_data = dataframe[
                    (dataframe[date_column] >= date_range_start) & (dataframe[date_column] <= date_range_end)
                ]

                # Display filtered data
                st.write(filtered_data)

                # Multiselect for column selection
                selected_columns = st.multiselect(f"Select Columns for {collection_name.capitalize()} Data",
                                                filtered_data.columns)
                if not selected_columns:
                    selected_columns = filtered_data.columns  # If none selected, download all columns

                # Download button for each DataFrame with selected columns
                download_button_str = f"Download {collection_name.capitalize()} Data"
                csv_data = filtered_data[selected_columns].to_csv(index=False).encode('utf-8')
                st.download_button(label=download_button_str, data=csv_data,
                                file_name=f"{collection_name}_data.csv", key=f"{collection_name}_data")

def Report():
    st.markdown("<h1 style='text-align: center;'>MissCallPay Report</h1>", unsafe_allow_html=True)
    # Sidebar
    st.sidebar.subheader("Filter Data by Date Range")
    dataset_date = pd.to_datetime("2023-02-27 01:26:38")
    date_range_start = st.sidebar.date_input("Start Date", dataset_date)
    date_range_end = st.sidebar.date_input("End Date", pd.to_datetime("today"))

    # Convert date_range_start and date_range_end to datetime objects
    date_range_start = pd.to_datetime(date_range_start).normalize()
    date_range_end = pd.to_datetime(date_range_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    # Retrieve data from MongoDB for each collection
    # Retrieve data from MongoDB for each collection
    for collection_name in collections:
        dataframes[collection_name] = retrieve_data_from_collection(db, collection_name)
        

        # Filter data based on date range
        date_column = None
        if collection_name == 'CDR':
            date_column = 'start_time'
        elif collection_name == 'ONBOARD':
            date_column = 'call_date'
        elif collection_name == 'USER ACTION':
            date_column = 'datetime'
        elif collection_name == 'USER LOG':
            date_column = 'call_date'
        elif collection_name == 'TRANSACTION':
            date_column = 'initiated_at'

        if date_column:
            dataframes[collection_name][date_column] = pd.to_datetime(dataframes[collection_name][date_column])
            filtered_data = dataframes[collection_name].loc[
                (dataframes[collection_name][date_column] >= date_range_start) &
                (dataframes[collection_name][date_column] <= date_range_end)
            ]
            dataframes[collection_name] = filtered_data
        
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Data Frame 1

    # User Try On-board
    action_data=dataframes['USER ACTION']
    try_user_onboard=action_data[(action_data['description'].isin([
        "Onboarding_Setted_Langauge_Success",
        "Onboarding_PostalPincode_Failure_No_Input",
        "Onboarding_Spoken_Bank_Name_Success",
        "Onboarding_Account_Selected_Failure",
        "Onboarding_Langauge_Selection_Failure_Noinput",
        "Onboarding_Account_Selected_Success",
        "Onboarding_Selected_Langauge_Failure",
        "Onboarding_PostalPincode_Failure_max_tries",
        "Onboarding_Spoken_Bank_Name_Failure",
        "Onboarding_PostalPincode_Success",
        "Onboarding_Selected_Langauge_MaxTries",
        "Onboarding_Account_Listing_Failure",
        "Onboarding_PostalPincode_Failure_invalid_input",
        "Onboarding_Spoken_Name_Failure",
        "Onboard_Shop_business_success",
        "Onboarding_Spoken_Name_Success",
        "Onboarding_Spoken_Customer_Name_Success",
        "Onboarding_Account_Listing_Success",
        "Onboarding_Last_Six_Digit_Failure_NoInput",
        "Onboarding_Last_Six_Digit_Failure_MaxTries",
        "Onboard_bank_otp_not_entered",
        "o2o_onboarding_failure",
        "Onboarding_Account_Selected",
        "Onboard_generate_otp_onboard_failed",
        "Onboarding_Langauge_Selection_Failure_MaxTries",
        "Onboarding_New_UPI_PIN_Failure",
        "Onboarding_Bank_OTP_Not_Entered_Failure",
        "Resume_continue",
        "Onboarding_Bank_OTP_MaxTries",
        "Onboard_new_upi_pin_invalid"
    ]))]['ID'].count()
    # User On Boarded Count
    Users_on_boarded=len(dataframes['ONBOARD'])
    # Successful trxns#-(IVR)
    txn_data = dataframes['TRANSACTION']
    ivr_count = txn_data[(txn_data['txn_request'].isin(['Collect Request', 'Send Money Request', 'Approve to pay']))&(txn_data['txn_type'].isin(['p2mm', 'p2p', 'p2em', 'p2e', 'p2m'])) 
                                & (txn_data['bank_txn_result'].isin(['Approved', 'Success']))]['txn_amount'].count()
    # Successful trxns#(Biller)
    biller_count=txn_data[ (txn_data['bill_pay_status'].isin(['completed']))&(txn_data['txn_type'].isin(['p2b']))]['txn_amount'].count()
    # Successful trxns#(QR Code)
    Qr_count = txn_data[(txn_data['txn_request'].isin(['QR', 'UPI APP'])) &(txn_data['txn_type'].isin(['p2mm', 'p2p', 'p2em', 'p2e', 'p2m']))
                                & (txn_data['bank_txn_result'].isin(['Approved', 'Success']))]['txn_amount'].count()
    # Successful trxns total #(IVR, QR Code, Biller)
    total_Scuccess_txn=txn_data[(txn_data['bank_txn_result'].isin(['Approved', 'Success']))]['txn_amount'].count()
    trxns_total = Qr_count + ivr_count + biller_count
    # Balance Check?
    userlog_data=dataframes['USER LOG']
    balance_check_count = userlog_data[userlog_data['lable_name'] == 'balance_cheked']['ID'].count()

    # Total Balance  check Select option_menu
    total_bc_count=userlog_data[userlog_data['lable_name'] == 'customer_selected_check_balance_option']['ID'].count()

    # Successful trxns value in Rupees (IVR)
    ivr_sum = txn_data[(txn_data['txn_request'].isin(['Collect Request', 'Send Money Request', 'Approve to pay']))
                                & (txn_data['bank_txn_result'].isin(['Approved', 'Success']))]['txn_amount'].sum()
    # Successful trxns value in Rupees (Biller)
    biller_sum=txn_data[ (txn_data['bill_pay_status'].isin(['completed']))]['txn_amount'].sum()

    Qr_sum = txn_data[(txn_data['txn_request'].isin(['QR', 'UPI APP']))
                                & (txn_data['bank_txn_result'].isin(['Approved', 'Success']))]['txn_amount'].sum()

    trxns_total_sum = Qr_sum + ivr_sum + biller_sum


    # TOTaL Column:

    t1=Users_on_boarded+2929
    t2=ivr_count+5462
    t3=biller_count+103
    t4=Qr_count+1576
    t5=t2+t3+t4
    t6=balance_check_count+1034
    t7=ivr_sum+709311.66
    t8=biller_sum+38589
    t9=Qr_sum+253281.25
    t10=t7+t8+t9




    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Data Frame 2

    # Failed Total Count
    Failed_total_count = txn_data[(~txn_data['bank_txn_result'].isin(['Approved', 'Success', 'EXPIRED', 'PENDING', '']))]['ID'].count()

    # Failed Transaction Count(QR Code & IVR)
    Failed_ivr_qr_count = txn_data[
        (txn_data['txn_type'].isin(['p2mm', 'p2p', 'p2em', 'p2e', 'p2m'])) &
        (~txn_data['bank_txn_result'].isin(['Approved', 'Success', 'EXPIRED', 'PENDING', '']))  # Exclude specified values and blank values
    ]['ID'].count()

    # Failed Transaction Count(Biller)
    Failed_biller_count = txn_data[
        (txn_data['txn_type'].isin(['p2b'])) &
        (~txn_data['bank_txn_result'].isin(['Approved', 'Success', 'EXPIRED', 'PENDING', '']))  # Exclude specified values and blank values
    ]['ID'].count()

    # Total Call Reached/Hitted
    data = dataframes['CDR']
    cdr_data = data[data['disposition'] != 'ANSWERED']
    inbound_count=cdr_data[ (cdr_data['call_direction'].isin(['in'])) ]['uid'].count()

    # Total Outbound Call
    outbound_count=cdr_data[ (cdr_data['call_direction'].isin(['out']))]['uid'].count()

    # Total Success Call (Outbound)
    success_call_count=cdr_data[ (cdr_data['call_direction'].isin(['out'])) & (cdr_data['disposition'].isin(['CALL_COMPLETED']))]['uid'].count()

    # Total Failed Call (Outbound)
    failed_call_count=cdr_data[ (cdr_data['call_direction'].isin(['out'])) & (cdr_data['disposition'].isin(['NO ANSWER','BUSY','FAILED']))]['uid'].count()
    # Highest Call Disconnected on<Leg_Name>
    action_data=dataframes['USER ACTION']
    most_common_description=''
    max_count=0
    if not action_data.empty:
        # Find the most duplicated value and its count
        most_common_description = action_data['description'].value_counts().idxmax()
        max_count = action_data['description'].value_counts().max()

        print(f"The most duplicated value in 'description' is: {most_common_description} with count: {max_count}")
    else:
        print("Filtered DataFrame is empty.")
    # most_common_description='Akash'
    # max_count=0

    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Report Designed 
    # 
    # Sample data
    # Format start and end dates
    formatted_start_date = date_range_start.strftime('%Y-%m-%d')
    formatted_end_date = (date_range_end - pd.Timedelta(days=1)).strftime('%Y-%m-%d')  # Subtract one day to get the end of the selected day
    data = {
        f'Particulars   {str(formatted_start_date)}:{str(formatted_end_date)}': ['Users on-boarded', 'Successful trxns(IVR)', 'Successful trxns(Biller)',
                        'Successful trxns(QR Code)', 'Successful trxns total(IVR, QR Code, Tata Power)',
                        'Balance Check', 'Successful trxns value in Rupees (IVR)',
                        'Successful trxns value in Rupees (Biller)', 'Successful trxns value in Rupees (QR Code)',
                        'Successful trxns value in Rupees total (IVR, QR Code, Tata Power)'],
        "Count":[Users_on_boarded, ivr_count, biller_count, Qr_count, trxns_total, balance_check_count, ivr_sum, biller_sum, Qr_sum, trxns_total_sum],
        "Persentage": [f"{round((Users_on_boarded / try_user_onboard) * 100, 2)}%", f"{round((ivr_count / total_Scuccess_txn) * 100, 2)}%", f"{round((biller_count / total_Scuccess_txn) * 100, 2)}%", 'N/A', 'N/A',f"{round((balance_check_count / total_bc_count) * 100, 2)}%" , 'N/A', 'N/A', 'N/A', 'N/A'],
        "Total": [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10]
    }

    # Create a DataFrame
    df1 = pd.DataFrame(data)


    data = {
        'Particulars': [
                        'Failed Transaction Count(QR Code & IVR)',
                        'Failed Transaction Count(Biller)',
                        'Total Call Reached/Hitted',
                        'Total Outbound Call',
                        'Total Success Call (Outbound)',
                        'Total Failed Call (Outbound)',
                        'Highest Call Disconnected on Leg Name'],
        "Count": [ Failed_ivr_qr_count, Failed_biller_count, inbound_count, outbound_count, success_call_count, failed_call_count, most_common_description],
        "Persentage": [f"{round((Failed_ivr_qr_count / Failed_total_count) * 100, 2)}%", f"{round((Failed_biller_count / Failed_total_count) * 100, 2)}%", '100%', f"{round((outbound_count / inbound_count) * 100, 2)}%", f"{round((success_call_count / outbound_count) * 100, 2)}%", f"{round((failed_call_count / outbound_count) * 100, 2)}%", f"{round((max_count / success_call_count) * 100, 2)}%"],
        "Total": [Failed_ivr_qr_count+1298, Failed_biller_count+84, inbound_count+31665, outbound_count+3000, success_call_count+21825, failed_call_count+8183, max_count+2813]
    }

    # Create a DataFrame
    df2 = pd.DataFrame(data)

    df1.index = np.arange(1, len(df1) + 1)
    df2.index = np.arange(1, len(df2) + 1)


    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Set the width and height for the DataFrame
    width = 1400

    # Designing
    st.dataframe(df1, width=width)  # No height parameter
    # Centered subheader using HTML
    st.markdown("<h3 style='text-align: center;'>Internal Purpose</h3>", unsafe_allow_html=True)
    st.dataframe(df2, width=width)  # No height parameter

    #___________________________________________________________________________________Download File______________________________________________

    # Function to generate a plot (image) with header and DataFrames
    def generate_plot(df1, df2):
        fig, ax = plt.subplots(2, 1, figsize=(15,7)) 

        # DataFrame 1
        ax[0].axis('off')
        ax[0].table(cellText=df1.values, colLabels=df1.columns, cellLoc='center', loc='center')
        ax[0].set_title( "MissCallPay Report", fontsize=20, ha='center', va='center')

        # DataFrame 2
        ax[1].axis('off')
        ax[1].table(cellText=df2.values, colLabels=df2.columns, cellLoc='center', loc='center')
        ax[1].set_title( "Internal Purpose", fontsize=16, ha='center', va='center')

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the plot to BytesIO
        image_buffer = BytesIO()
        plt.savefig(image_buffer, format='png')
        image_buffer.seek(0)

        return image_buffer

    # Download button for the generated image
    if st.button("Download Report"):
        # Show pop-up message before initiating download
        st.warning("Before sharing data, please take approval from your superior.")
        download_button = st.download_button(
            label="Download Report",
            key="combined_image_download",
            data=generate_plot(df1, df2),
            file_name="MissCallPay_Report.png",
            mime="image/png"
        )
        st.write(download_button)
def Dahboard():
    st.markdown("<h1 style='text-align: center;'>MissCallPay Dashboard</h1>", unsafe_allow_html=True)


    for collection_name in collections:
        dataframes[collection_name] = retrieve_data_from_collection(db, collection_name)

    # Dashboard
    cdr_data = dataframes['CDR']

    # Convert 'start_time' to datetime type
    cdr_data['start_time'] = pd.to_datetime(cdr_data['start_time'])

    # Extract hour from 'start_time'
    cdr_data['hour'] = cdr_data['start_time'].dt.hour

    # Count incoming and outbound calls hour-wise
    hourly_calls_count = cdr_data.groupby(['hour', 'call_direction']).size().reset_index(name='count')

    # Pivot the dataframe for plotting
    pivot_df = hourly_calls_count.pivot(index='hour', columns='call_direction', values='count').fillna(0).reset_index()

    # Create clustered column chart using Plotly Express
    fig_hourly_calls = go.Figure()

    # Add traces
    fig_hourly_calls.add_trace(go.Bar(x=pivot_df['hour'], y=pivot_df['in'], name='Incoming Calls'))
    fig_hourly_calls.add_trace(go.Bar(x=pivot_df['hour'], y=pivot_df['out'], name='Outbound Calls'))

    # Update layout
    fig_hourly_calls.update_layout(
        title='Hourly Count of Incoming and Outbound Calls',
        xaxis=dict(title='Hour'),
        yaxis=dict(title='Count'),
        barmode='group',
        template='plotly_dark',
        width=670,
        height=500
    )

    # Total Success Call (Outbound)
    success_call_count = cdr_data[(cdr_data['call_direction'].isin(['out'])) & (cdr_data['disposition'].isin(['CALL_COMPLETED']))]['uid'].count()

    # Total Failed Call (Outbound)
    failed_call_count = cdr_data[(cdr_data['call_direction'].isin(['out'])) & (cdr_data['disposition'].isin(['NO ANSWER', 'BUSY', 'FAILED']))]['uid'].count()

    # Create a DataFrame for the bar chart
    calls_data = pd.DataFrame({
        'Call Type': ['Answer Calls', 'Failed Calls'],
        'Count': [success_call_count, failed_call_count]
    })

    # Create a bar chart using Plotly Express
    fig_calls = px.bar(
        calls_data,
        x='Call Type',
        y='Count',
        color='Call Type',
        labels={'Count': 'Number of Calls'},
        title='Count Of Success Calls & Failed Calls',
        template='plotly_dark',
        width=670,
        height=500
    )
    ################################################
    # Action data
    action_data = dataframes['USER ACTION']

    # Filter out rows with missing or null descriptions
    action_data = action_data[action_data['description'].notna()]

    # Get the top 10 descriptions and their counts
    top_descriptions = action_data['description'].value_counts().nlargest(10)
    top_descriptions_data = pd.DataFrame({
        'Description': top_descriptions.index,
        'Count': top_descriptions.values
    })

    # Create a bar chart using Plotly Express
    fig_top_reasons = px.bar(
        top_descriptions_data,
        x='Count',  # Change 'Description' to 'Count' for better clarity
        y='Description',
        orientation='h',  # Change to horizontal orientation
        color='Description',
        labels={'Count': 'Number of Occurrences', 'Description': 'Disconnected Reasons'},  # Update axis labels
        title='Top 10 Last Disconnected Reasons',
        template='plotly_dark',
        width=1400,
        height=500
    )

    # Modify the appearance of the chart
    fig_top_reasons.update_layout(
        xaxis=dict(title='Number of Occurrences'),
        yaxis=dict(title='Disconnected Reasons'),
        showlegend=False,  # Remove the legend for better clarity
    )


    #############################################################################################################################
    #Onboard Data

    # Date Wise Onboard User
    # Retrieve data from the 'onboard' collection
    onboard_data =dataframes['ONBOARD']

    # Convert 'call_date' to datetime type
    onboard_data['call_date'] = pd.to_datetime(onboard_data['call_date'])

    # Extract day from 'call_date'
    onboard_data['day'] = onboard_data['call_date'].dt.day

    # Count onboarded users day-wise
    daily_onboard_count = onboard_data.groupby('day').size().reset_index(name='count')

    # Create clustered column chart using Plotly Express
    fig_onboard_count = px.bar(
        daily_onboard_count,
        x='day',
        y='count',
        title='Onboarded Users Count By Date',
        labels={'count': 'Onboarded Users Count', 'day': 'Day'},
        template='plotly_dark',
        width=1400,
        height=500
    )

    # Update layout
    fig_onboard_count.update_layout(xaxis=dict(title='Day'))

    #############################################
    #Total Onboarded Customer Count By Pages Wise
    did_data=dataframes['did_details']

    # Convert column names to lowercase for case-insensitivity
    onboard_data.columns = onboard_data.columns.str.lower()
    did_data.columns = did_data.columns.str.lower()

    # Merge 'did' and 'onboard' data on the 'did' column
    merged_data = pd.merge(did_data, onboard_data, on='did', how='inner')

    # Count onboarded users by page
    page_wise_count = merged_data['page'].value_counts().reset_index()
    page_wise_count.columns = ['page', 'count']

    # Create a bar chart using Plotly Express
    fig_page_wise_bar = px.bar(
        page_wise_count,
        x='page',
        y='count',
        color='page',
        labels={'count': 'Number of Onboarded Users', 'page': 'Page'},
        title='Page-wise Count of Onboarded Users',
        template='plotly_dark',
        width=1400,
        height=500
    )
    ###########################################################

    # Trasaction Data

    #Transactions by Date

    # Retrieve transaction data
    txn_data = dataframes['TRANSACTION']
    # # Convert 'initiated_at' to datetime type
    # txn_data['initiated_at'] = pd.to_datetime(txn_data['initiated_at'])

    # # Extract day from 'initiated_at'
    # txn_data['day'] = txn_data['initiated_at'].dt.day

    # # Count successful and failed transactions day-wise
    # daily_txn_count = txn_data.groupby(['day', 'bank_txn_result']).size().reset_index(name='count')

    # # Pivot the dataframe for plotting
    # pivot_txn_df = daily_txn_count.pivot(index='day', columns='bank_txn_result', values='count').fillna(0).reset_index()

    # # Create clustered column chart using Plotly Express
    # fig_txn_status = go.Figure()

    # # Add traces
    # fig_txn_status.add_trace(go.Bar(x=pivot_txn_df['day'], y=pivot_txn_df[('Approved', 'Success')], name='Successful Transactions'))
    # fig_txn_status.add_trace(go.Bar(x=pivot_txn_df['day'], y=pivot_txn_df[('EXPIRED', 'PENDING')], name='Failed Transactions'))

    # # Update layout
    # fig_txn_status.update_layout(
    #     title='Daily Count of Successful and Failed Transactions',
    #     xaxis=dict(title='Day'),
    #     yaxis=dict(title='Count'),
    #     barmode='group',
    #     template='plotly_dark',
    #     width=670,
    #     height=500
    # )

    # # Show the plot
    # fig_txn_status.show()

    ##################################
    #Proportion Of Successful Payment And Failed Payment
    # Failed Total Count
    # Calculate amounts
    failed_total_amount =  txn_data[(~txn_data['bank_txn_result'].isin(['Approved', 'Success', 'EXPIRED', 'PENDING', '']))]['txn_amount'].sum() 
    success_total_amount = txn_data[(txn_data['bank_txn_result'].isin(['Approved', 'Success']))]['txn_amount'].sum()

    # Create a pie chart using Plotly Express
    fig_txn_amount_pie = px.pie(
        names=['Failed Transactions', 'Successful Transactions'],
        values=[failed_total_amount, success_total_amount],
        title='Total Transaction Amount Overview',
        template='plotly_dark',
        width=670,
        height=500
    )
    failed_total_amount =  txn_data[(~txn_data['bank_txn_result'].isin(['Approved', 'Success', 'EXPIRED', 'PENDING', '']))]['txn_amount'].count() 
    success_total_amount = txn_data[(txn_data['bank_txn_result'].isin(['Approved', 'Success']))]['txn_amount'].count()

    # Create a pie chart using Plotly Express
    fig_txn_count_pie = px.pie(
        names=['Failed Transactions', 'Successful Transactions'],
        values=[failed_total_amount, success_total_amount],
        title='Total Successes And Failed Transaction Count',
        template='plotly_dark',
        width=670,
        height=500
    )
    ##################################
    # Count and percentage of failed transactions
    failed_frame = pd.DataFrame(txn_data[~txn_data['bank_txn_result'].isin(['Approved', 'Success', 'EXPIRED', 'PENDING', ''])])
    # Get the top 10 descriptions and their counts
    top_bank_txn_reasons = failed_frame['bank_txn_result'].value_counts()
    top_bank_txn_reasons_data = pd.DataFrame({
        'Reasons': top_bank_txn_reasons.index,
        'Count': top_bank_txn_reasons.values
    })

    # Create a bar chart using Plotly Express
    fig_dis_reasons = px.bar(
        top_bank_txn_reasons_data,
        x='Count',  # Change 'Description' to 'Count' for better clarity
        y='Reasons',
        orientation='h',  # Change to horizontal orientation
    
        labels={'Count': 'Number of Occurrences', 'Reasons': 'Bank Transaction Reasons'},  # Update axis labels
        title='Top 10 Last Disconnected Reasons',
        template='plotly_dark',
        width=1400,
        height=500
    )

    # Modify the appearance of the chart
    fig_top_reasons.update_layout(
        xaxis=dict(title='Number of Occurrences'),
        yaxis=dict(title='Transaction Failed Reasons'),
        showlegend=False,  # Remove the legend for better clarity
    )

    # Load Style css
    with open('style.css')as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Theme
    hide_st_style = """
    <style>
    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    header {visibility:hidden;}
    </style>
    """

    # Arrange charts side by side
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_hourly_calls)
    col2.plotly_chart(fig_calls)
    # Show the chart
    st.plotly_chart(fig_top_reasons)
    # Show the chart
    st.plotly_chart(fig_onboard_count)

    # Show the Bar Chart
    st.plotly_chart(fig_page_wise_bar)
    # Show the chart
    # st.plotly_chart(fig_txn_status)
    # Show the chart
    col3, col4 = st.columns(2)
    col3.plotly_chart(fig_txn_count_pie)
    col4.plotly_chart(fig_txn_amount_pie)

    ####
    st.plotly_chart(fig_dis_reasons)
    # Hide elements
    st.markdown(hide_st_style, unsafe_allow_html=True)

def creds_entered():
    if not st.session_state["user"]:
        st.warning("Please enter username.")
        return

    if not st.session_state["passwd"]:
        st.warning("Please enter password.")
        return

    if st.session_state["user"].strip() == "admin" and st.session_state["passwd"].strip() == "pass":
        st.session_state["authenticated"] = True
        st.session_state["user_role"] = "admin"
    elif st.session_state["user"].strip() == "user" and st.session_state["passwd"].strip() == "pass":
        st.session_state["authenticated"] = True
        st.session_state["user_role"] = "user"
    else:
        st.session_state["authenticated"] = False
        st.error("Invalid Username/Password :face_with_raised_eyebrow:")

def authenticate():
    
    st.markdown("<h1 style='text-align: center;'>MISSCALLPAY AUTOMATION TOOL</h1>", unsafe_allow_html=True)

    # Adjusting logo size
    # logo_path = r"C:\Users\miv-003\OneDrive\Automation Dashboard\data\logo.png"
    # logo = Image.open(logo_path)

    # Displaying logo without changing its size
    # st.image(logo, use_column_width=False, output_format="auto")

    if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
        col1, col2, col3 = st.columns(3)

        # Centering the form
        with col2:
            st.text_input(label="Username:", key="user", on_change=creds_entered)
            st.text_input(label="Password", key="passwd", type="password", on_change=creds_entered)
            login_button = st.button("Login")
        
        if login_button:
            creds_entered()

        return False

    if st.session_state["authenticated"]:
        return True
    else:
        return False


def logout():
    st.experimental_set_query_params()
    st.session_state["authenticated"] = False
    st.session_state["user_role"] = None
    st.success("Logout successful!")
    main()  # Redirect to the main function

def sideBar():
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=["Home", "Data", "Report", "Dashboard", "Logout"],  # Fix the typo here
            icons=["house", "eye", "eye", "eye", "eye"],
            menu_icon="cast",
            default_index=0
        )
    if selected == "Home":
        Home()
    elif selected == "Data":
        Data_show()
    elif selected == "Report":
        Report()
    elif selected == "Dashboard":  # Fix the function name here
        Dahboard()
    elif selected == "Logout":
        logout()

def main():
    if authenticate():
        sideBar()

if __name__ == "__main__":
    main()

# load Style css
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)




#theme
hide_st_style=""" 

<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
</style>
"""

st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)

       
