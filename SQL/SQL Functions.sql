-- Step 1: Create the sales_data table
CREATE TABLE sales_data (
    sale_id INT PRIMARY KEY,
    product VARCHAR(50),
    category VARCHAR(50),
    quantity INT,
    price DECIMAL(10, 2),
    customer_id VARCHAR(10),
    region VARCHAR(50)
);

-- Step 2: Insert the data into the table
INSERT INTO sales_data (sale_id, product, category, quantity, price, customer_id, region) VALUES
(1, 'Laptop', 'Electronics', 2, 1200, 'C001', 'East'),
(2, 'Smartphone', 'Electronics', 1, 800, 'C002', 'West'),
(3, 'Book', 'Books', 5, 20, 'C003', 'North'),
(4, 'Headphones', 'Electronics', 3, 150, 'C001', 'East'),
(5, 'Notebook', 'Books', 10, 5, 'C004', 'South'),
(6, 'Smartphone', 'Electronics', 2, 800, 'C002', 'West');

-- SELECT Total items
SELECT COUNT(*) AS total_items
FROM sales_data;

-- Get the total company sales
SELECT SUM(quantity*price) AS total_sales
FROM sales_data;

-- Average price from the sales_data table
SELECT AVG(price) AS avg_price
FROM sales_data;

-- Average quanntity from sales_data table
SELECT AVG(quantity) AS avg_quantity
FROM sales_data;

-- Select the most expensive
SELECT product,MAX(price) AS most_expensive
FROM sales_data;

-- Select the cheapest product
SELECT product, MIN(price) AS cheapest_product
FROM sales_data;

-- How many items of each category were sold
SELECT category, SUM(quantity) AS total_items
FROM sales_data
GROUP BY category;


-- Get the total sales per category
SELECT category, SUM(quantity*price) AS total_sales
FROM sales_data
GROUP BY category;
