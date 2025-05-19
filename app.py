from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import numpy as np
import plotly.express as px
import pandas as pd  # Import pandas for handling CSV
import plotly.io as pio
import subprocess

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load models
with open('models/sales_model1.pkl2', 'rb') as f:
    sales_model = pickle.load(f)

with open('models/profit_model2.pkl', 'rb') as f:
    profit_model = pickle.load(f)

with open('models/label_encoders2.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Home Page (Login)
@app.route('/', methods=['GET', 'POST'])
def home():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin':
            session['user'] = username
            return redirect(url_for('predict'))
        else:
            error = 'Invalid Credentials'
    return render_template('home.html', error=error)

# Prediction Page
@app.route('/predict', methods=['GET', 'POST'])
def predict():

    predicted_sales = None
    profit_prediction = None

    if request.method == 'POST':
        try:
            # Get user input
            category_name = request.form['category_name']
            customer_region = request.form['customer_region']
            shipping_type = request.form['shipping_type']
            order_quantity = int(request.form['order_quantity'])
            days_scheduled = int(request.form['days_for_shipment_scheduled'])

            # Encode inputs
            cat = encoders['category_name'].transform([category_name])[0]
            region = encoders['customer_region'].transform([customer_region])[0]
            shipping = encoders['shipping_type'].transform([shipping_type])[0]

            input_data = np.array([[cat, region, shipping, order_quantity, days_scheduled]])

            # Predict sales
            predicted_sales = round(sales_model.predict(input_data)[0], 2)

            # Predict profit class
            profit_class = profit_model.predict(input_data)[0]
            profit_prediction = "Profitable ✅" if profit_class == 1 else "Not Profitable ❌"
        except Exception as e:
            predicted_sales = None
            profit_prediction = f"Error: {str(e)}"

    return render_template('predict.html',
                           predicted_sales=predicted_sales,
                           profit_prediction=profit_prediction)
@app.route('/visual', methods=['GET', 'POST'])
def chart():
    selected_region = request.form.get('region')
    selected_category = request.form.get('category')
    df = pd.read_csv("Ecommerce_data.csv", encoding='latin-1')

    # Apply filters if selected
    filtered_df = df.copy()
    if selected_region and selected_region != "All":
        filtered_df = filtered_df[filtered_df["customer_region"] == selected_region]
    if selected_category and selected_category != "All":
        filtered_df = filtered_df[filtered_df["category_name"] == selected_category]

    # Aggregate for bar chart: Total Sales by Category
    bar_data = filtered_df.groupby("category_name")["sales_per_order"].sum().reset_index()

    # Plotly bar chart
    fig = px.bar(
        bar_data,
        x="category_name",
        y="sales_per_order",
        title="Total Sales by Category (Filtered)",
        labels={"sales_per_order": "Total Sales", "category_name": "Category"},
        color="category_name"
    )
    fig.update_layout(xaxis_tickangle=-45)

    graph_html = pio.to_html(fig, full_html=False)

    # Dropdown values
    regions = ["All"] + sorted(df["customer_region"].dropna().unique())
    categories = ["All"] + sorted(df["category_name"].dropna().unique())

    return render_template("chart_filter.html", chart=graph_html, regions=regions, categories=categories,
                           selected_region=selected_region, selected_category=selected_category)


@app.route('/charts', methods=['GET', 'POST'])
def charts():  # ✅ Renamed function to avoid conflict
    df = pd.read_csv("Ecommerce_data.csv", encoding='latin-1')

    selected_region = request.form.get('region', 'All')
    selected_category = request.form.get('category', 'All')
    selected_chart = request.form.get('chart_type', 'Bar')

    if selected_region != 'All':
        df = df[df['customer_city'] == selected_region]

    if selected_category != 'All':
        df = df[df['category_name'] == selected_category]

    # Group by city and category for bar charts
    bar_data = (
        df.groupby(["customer_city", "category_name"])["sales_per_order"]
        .sum()
        .reset_index()
    )

    # Get top 10 cities by total sales
    top_cities = (
        bar_data.groupby("customer_city")["sales_per_order"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .index.tolist()
    )
    bar_data = bar_data[bar_data["customer_city"].isin(top_cities)]

    # Chart rendering logic
    if selected_chart == 'Bar':
        fig = px.bar(
            bar_data,
            x="customer_city",
            y="sales_per_order",
            color="category_name",
            title="Top 10 Cities by Sales (Grouped by Category)",
            labels={"sales_per_order": "Total Sales", "customer_city": "City", "category_name": "Category"},
            barmode='group'
        )
    elif selected_chart == 'Pie':
        pie_data = (
            df.groupby("category_name")["sales_per_order"]
            .sum()
            .reset_index()
        )
        fig = px.pie(
            pie_data,
            names="category_name",
            values="sales_per_order",
            title="Sales Distribution by Category"
        )
    elif selected_chart == 'Line':
        line_data = (
            df.groupby(["customer_city", "category_name"])["sales_per_order"]
            .sum()
            .reset_index()
        )
        fig = px.line(
            line_data[line_data["customer_city"].isin(top_cities)],
            x="customer_city",
            y="sales_per_order",
            color="category_name",
            title="Sales Trend by City and Category",
            markers=True
        )

    fig.update_layout(xaxis_tickangle=-45)
    graph_html = fig.to_html(full_html=False)

    city_list = ['All'] + sorted(df['customer_city'].dropna().unique().tolist())
    category_list = ['All'] + sorted(df['category_name'].dropna().unique().tolist())

    return render_template('index.html',
                           graph_html=graph_html,
                           city_list=city_list,
                           category_list=category_list,
                           selected_region=selected_region,
                           selected_category=selected_category,
                           selected_chart=selected_chart)

# Power BI paths
POWER_BI_PATH = r"C:\Program Files\Microsoft Power BI Desktop\bin\PBIDesktop.exe"
PBIX_FILE = r"C:\Users\YourName\Path\To\Ecommerce.pbix"

  # You’ll create this next


@app.route('/open-powerbi')
def open_powerbi():
    # Check if the user is logged in before accessing this route
    # Redirect to login if the user is not logged in

    try:
        # Try to open the Power BI file using subprocess
        subprocess.Popen([POWER_BI_PATH, PBIX_FILE])
        # Render the template with a success message and a logout button
        return render_template('dashboard.html', message="✅ Power BI Dashboard is opening!")
    except Exception as e:
        # In case of an error, return the error message
        return f"❌ Error: {e}"

# Logout
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)
