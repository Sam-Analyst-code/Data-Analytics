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