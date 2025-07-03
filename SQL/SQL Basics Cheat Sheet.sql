-- CREATE A DATABASE hermira
CREATE DATABASE hermira;
USE hermira;

-- CREATE A sales TABLE (id, name, price, quantity, date)
CREATE TABLE sales (
    id INT PRIMARY KEY,
    product_name VARCHAR(255) NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    quantity INT NOT NULL,
    date DATE NOT NULL,
    category_id INT NOT NULL,
    FOREIGN KEY (category_id) REFERENCES category(id)
);

-- CREATE A category TABLE (id, name)
CREATE TABLE category (
    id INT PRIMARY KEY,
    category_name VARCHAR(255) NOT NULL
);

-- INSERT DATA INTO THE category TABLE
INSERT INTO category (id, category_name)
VALUES
(1, 'Electronics'),
(2, 'Home Appliances'),
(3, 'Books'),
(4, 'Clothing'),
(5, 'Toys');

-- INSERT DATA INTO THE sales TABLE (30 rows)
INSERT INTO sales (id, product_name, price, quantity, date, category_id)
VALUES
(1, 'Smartphone', 699.99, 10, '2025-01-01', 1),
(2, 'Laptop', 999.99, 5, '2025-01-02', 1),
(3, 'Headphones', 199.99, 15, '2025-01-03', 1),
(4, 'Microwave', 89.99, 8, '2025-01-04', 2),
(5, 'Refrigerator', 499.99, 3, '2025-01-05', 2),
(6, 'Washing Machine', 399.99, 4, '2025-01-06', 2),
(7, 'Fiction Book', 14.99, 20, '2025-01-07', 3),
(8, 'Cookbook', 24.99, 10, '2025-01-08', 3),
(9, 'Textbook', 59.99, 7, '2025-01-09', 3),
(10, 'T-Shirt', 19.99, 25, '2025-01-10', 4),
(11, 'Jeans', 49.99, 12, '2025-01-11', 4),
(12, 'Jacket', 89.99, 6, '2025-01-12', 4),
(13, 'Action Figure', 29.99, 18, '2025-01-13', 5),
(14, 'Board Game', 39.99, 9, '2025-01-14', 5),
(15, 'Doll', 19.99, 14, '2025-01-15', 5),
(16, 'Tablet', 299.99, 7, '2025-01-16', 1),
(17, 'Smartwatch', 199.99, 10, '2025-01-17', 1),
(18, 'Bluetooth Speaker', 49.99, 20, '2025-01-18', 1),
(19, 'Vacuum Cleaner', 149.99, 5, '2025-01-19', 2),
(20, 'Air Conditioner', 599.99, 2, '2025-01-20', 2),
(21, 'Self-Help Book', 19.99, 15, '2025-01-21', 3),
(22, 'Biography', 24.99, 8, '2025-01-22', 3),
(23, 'Sweater', 29.99, 10, '2025-01-23', 4),
(24, 'Shoes', 79.99, 6, '2025-01-24', 4),
(25, 'Puzzle', 14.99, 12, '2025-01-25', 5),
(26, 'Drone', 499.99, 3, '2025-01-26', 1),
(27, 'Camera', 899.99, 4, '2025-01-27', 1),
(28, 'Blender', 59.99, 10, '2025-01-28', 2),
(29, 'Notebook', 9.99, 30, '2025-01-29', 3),
(30, 'Scarf', 14.99, 20, '2025-01-30', 4);

-- SELECT STATEMENT

SELECT product_name, price, quantity
FROM sales
WHERE price > 100;

-- SELECT ONE COLUMN
SELECT product_name
FROM sales;

-- SELECT MULTIPLE COLUMN
-- You can select multiple columns by separating them with commas.
-- This allows you to retrieve specific data from the table.
SELECT product_name, price, quantity
FROM sales;

--SELECT ALL COLUMNS
-- The asterisk (*) is used to select all columns from the table.
-- This is useful when you want to retrieve all data without specifying each column name.
SELECT *
FROM sales;

-- USING ALIAS
-- Aliases are used to give a temporary name to a column or table in a query.
-- This makes the output more readable and easier to understand.
SELECT product_name AS name, price AS cost, quantity AS amount
FROM sales;

-- USING DISTINCT
-- Aliases are used to give a temporary name to a column or table in a query.
-- This makes the output more readable and easier to understand.
SELECT product_name AS name, price AS cost, quantity AS amount
FROM sales;


-- USING DISTINCT
-- The DISTINCT keyword is used to return only unique values in a column.
-- This is helpful when you want to eliminate duplicate entries from the result set.
SELECT DISTINCT category_id
FROM sales;



-- CONCAT FUNCTION

-- The CONCAT function is used to combine two or more strings into a single string.
-- It is useful when you want to create a combined output from multiple columns or values.

-- Example: Combine product_name and price into a single string.
SELECT CONCAT(product_name, ' - $', price) AS product_details
FROM sales;



-- WHERE STATEMENT

-- The WHERE clause is used to filter records based on a specified condition.
-- Example: Select all products with a quantity greater than 10.

SELECT product_name, quantity
FROM sales
WHERE quantity > 10;

-- Example: Select all products from the 'Electronics' category (category_id = 1).

SELECT product_name, category_id
FROM sales
WHERE category_id = 1;

-- ORDER BY STATEMENT
-- The ORDER BY clause is used to sort the result set in ascending or descending order.
-- By default, it sorts in ascending order.
-- Example: Select all products and order them by price in ascending order.
SELECT product_name, price
FROM sales
ORDER BY price ASC;

-- Functions and Aggregations

-- Functions and aggregations are used to perform calculations on data in a table.
-- They help summarize and analyze data effectively.

-- COUNT FUNCTION
-- The COUNT function is used to count the number of rows that match a specified condition.
-- Example: Count the total number of products in the sales table.
SELECT COUNT(*) AS total_products
FROM sales;

-- SUM FUNCTION
-- The SUM function is used to calculate the total sum of a numeric column.
-- Example: Calculate the total sales amount (price * quantity) for all products.
SELECT SUM(price * quantity) AS total_sales
FROM sales;

-- Returns the total sales amount for all products in the sales table.

--AVG FUNCTION
-- The AVG function is used to calculate the average value of a numeric column.
SELECT AVG(price) AS average_price
FROM sales;

-- Returns the average price of all products in the sales table.


-- MAX FUNCTION
-- The MAX function is used to find the maximum value in a numeric column.

SELECT MAX(price) AS highest_price
FROM sales;
-- Returns the highest price among all products in the sales table.

-- MIN FUNCTION
-- The MIN function is used to find the minimum value in a numeric column.

SELECT MIN(price) AS lowest_price
FROM sales;
-- Returns the lowest price among all products in the sales table.

-- GROUP BY STATEMENT
-- The GROUP BY clause is used to group rows that have the same values in specified columns into summary rows.
-- It is often used with aggregate functions like COUNT, SUM, AVG, etc.
-- Example: Count the number of products in each category.

SELECT category_id, COUNT(*) AS total_products
FROM sales
GROUP BY category_id;
-- Returns the total number of products in each category.

-- HAVING STATEMENT
-- The HAVING clause is used to filter records after the GROUP BY operation.
-- It is similar to the WHERE clause but is used with aggregate functions.
-- Example: Count the number of products in each category and filter categories with more than 5 products.

SELECT category_id, COUNT(*) AS total_products
FROM sales
GROUP BY category_id
HAVING COUNT(*) > 5;
-- Returns the total number of products in each category, but only for categories with more than 5 products.

