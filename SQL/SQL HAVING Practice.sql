-- Step 1: Create the sales_data table
CREATE TABLE sales_data (
    id INT PRIMARY KEY,
    product VARCHAR(50),
    category VARCHAR(50),
    store VARCHAR(50),
    quantity INT,
    revenue INT
);

-- Step 2: Insert the provided data into the sales_data table
INSERT INTO sales_data (id, product, category, store, quantity, revenue) VALUES
(1, 'Bread', 'Bakery', 'Nairobi', 20, 800),
(2, 'Milk', 'Dairy', 'Mombasa', 30, 1200),
(3, 'Bread', 'Bakery', 'Mombasa', 10, 400),
(4, 'Cheese', 'Dairy', 'Nairobi', 25, 2500),
(5, 'Cake', 'Bakery', 'Nairobi', 5, 1000),
(6, 'Yogurt', 'Dairy', 'Nairobi', 15, 900);



SELECT category, SUM(revenue)
FROM sales_data
GROUP BY category;

-- SELECT categories whose revenue is above/greater 3000
SELECT category, SUM(revenue)
FROM sales_data
GROUP BY category
HAVING SUM(revenue) >3000;

-- Show stores with more than 2 products
SELECT store, COUNT(DISTINCT product) AS total_products
FROM sales_data
GROUP BY store
HAVING COUNT(product) > 2;


-- Select categories with less than 40 quantities sold
SELECT category,SUM(quantity)
FROM sales_data
GROUP BY category
HAVING SUM(quantity)< 40;

-- Find categories where the average quantity sold is greater than 15
SELECT category, AVG(quantity) as avg_quantity
FROM sales_data
GROUP BY category
HAVING AVG(quantity) > 15;