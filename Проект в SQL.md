В этом проекте рассмотрены данные о продажах магазина, включающие расходы и сегментацию покупателей, продажи по регионам, показатели рентабельности и др. Анализируем данные в PostgreSQL.

Берем данные из таблиц [orders.csv](https://github.com/UlyanaNovikova/Data-analytics/blob/main/Files/orders.csv), [returns.csv](https://github.com/UlyanaNovikova/Data-analytics/blob/main/Files/returns.csv). 


### Содержание
- [Базовый анализ данных](#базовый-анализ-данных)
- [Группировка данных](#группировка-данных)
- [Join](#join)
- [Where, Having, Case](#where-having-case)
- [Оконные функции](#оконные-функции)


### Базовый анализ данных
Выведем номер заказа и категорию товара для определенного клиента
```
SELECT order_id, category
FROM orders
WHERE Customer_ID = 'SO-20335'
LIMIT 5;


order_id      |category       |
--------------+---------------+
US-2017-108966|Furniture      |
US-2017-108966|Office Supplies|
CA-2017-161718|Furniture      |
CA-2017-161718|Furniture      |
CA-2017-161718|Technology     |
```


Найдем общее количество заказов, оформленных в каждом регионе

```
SELECT Region, COUNT(*) AS Total_Orders
FROM orders
GROUP BY Region;


region |total_orders|
-------+------------+
South  |        1620|
West   |        3203|
East   |        2848|
Central|        2323|
```


### Группировка данных
Найдем топ-5 покупателей с наибольшим общим объемом продаж
```
SELECT Customer_ID, ROUND(SUM(Sales),1) AS Total_Sales
FROM orders
GROUP BY Customer_ID
ORDER BY Total_Sales DESC
LIMIT 5;


customer_id|total_sales|
-----------+-----------+
SM-20320   |    25043.1|
TC-20980   |    19052.2|
RB-19360   |    15117.3|
TA-21385   |    14595.6|
AB-10105   |    14473.6|
```

Определим среднюю маржу прибыли для каждой категории товаров
```
SELECT Category, ROUND(AVG(Profit / Sales),2) AS Avg_Profit_Margin
FROM orders
GROUP BY Category;


category       |avg_profit_margin|
---------------+-----------------+
Furniture      |             0.04|
Office Supplies|             0.14|
Technology     |             0.16|
```


### Join
Объединим таблицу "Orders" с таблицей "Returns", чтобы найти возвращенные заказы
```
SELECT o.order_id, r.returned
FROM orders o
LEFT JOIN returns r ON o.Order_ID = r.Order_ID
WHERE r.returned='Yes'
LIMIT 5;


order_id      |returned|
--------------+--------+
CA-2016-143336|Yes     |
CA-2016-143336|Yes     |
CA-2016-143336|Yes     |
CA-2016-143336|Yes     |
CA-2016-143336|Yes     |
```


### Where, Having, Case
Найдем товары с отрицательной прибылью
```
SELECT product_name, ROUND(Profit,1) AS Profit
FROM orders
WHERE Profit < 0
ORDER BY ABS(Profit) DESC
LIMIT 5;


product_name                             |profit |
-----------------------------------------+-------+
Cubify CubeX 3D Printer Double Head Print|-6600.0|
Cubify CubeX 3D Printer Triple Head Print|-3840.0|
GBC DocuBind P400 Electric Binding System|-3701.9|
Lexmark MX611dhe Monochrome Laser Printer|-3400.0|
Ibico EPK-21 Electric Binding System     |-2929.5|
```


Найдем заказы с суммой продаж более 10000 долларов и скидкой более 10 %
```
SELECT order_id, ROUND(sales,1) AS Sales, discount
FROM orders
WHERE Sales > 10000 AND Discount > 0.10;


order_id      |sales  |discount|
--------------+-------+--------+
CA-2019-127180|11200.0|    0.20|
CA-2016-145317|22638.5|    0.50|
```

Найдем топ-5 заказов по продажам в декабре 2018 года
```
SELECT order_id, order_date, ROUND(sales,1) AS Sales
FROM orders
WHERE EXTRACT(YEAR FROM Order_Date) = 2018 AND EXTRACT(MONTH FROM Order_Date) = 12
ORDER BY sales DESC
LIMIT 5;


order_id      |order_date|sales |
--------------+----------+------+
CA-2018-117121|2018-12-17|9892.7|
US-2018-116729|2018-12-25|2575.9|
CA-2018-169670|2018-12-25|2563.1|
CA-2018-143805|2018-12-01|2104.6|
CA-2018-163804|2018-12-02|2079.4|
```

Определим клиентов со средней суммой продаж более 1500 долларов
```
SELECT Customer_ID, ROUND(AVG(Sales),1) AS Avg_Sales
FROM orders
GROUP BY Customer_ID
HAVING AVG(Sales) > 1500
ORDER BY Avg_Sales DESC;


customer_id|avg_sales|
-----------+---------+
MW-18235   |   1751.3|
SM-20320   |   1669.5|
TC-20980   |   1587.7|
GT-14635   |   1558.5|
```

Найдем регионы, в которых объем продаж выше среднего объема продаж по всем регионам
```
SELECT Region, ROUND(SUM(Sales),1) AS Total_Sales
FROM orders
GROUP BY Region
HAVING SUM(Sales) > (SELECT AVG(Total_Sales) FROM (SELECT SUM(Sales) AS Total_Sales FROM orders GROUP BY Region) AS avg_sales_per_region);


region|total_sales|
------+-----------+
West  |   725457.8|
East  |   678781.2|
```

Найдем топ-5 подкатегорий со стандартным отклонением продаж больше 500
```
SELECT subcategory, ROUND(STDDEV(Sales),1) AS Sales_SD, ROUND(avg(Sales),1) AS Sales_AVG
FROM orders
GROUP BY subcategory
HAVING STDDEV(Sales) > 500
ORDER BY Sales_SD DESC
LIMIT 5;


subcategory|sales_sd|sales_avg|
-----------+--------+---------+
Copiers    |  3175.7|   2198.9|
Machines   |  2765.1|   1645.6|
Supplies   |   923.8|    245.7|
Bookcases  |   638.7|    503.9|
Tables     |   615.8|    648.8|
```

Разделим клиентов на категории в зависимости от общей суммы продаж
```
SELECT Customer_ID,
       ROUND(SUM(Sales),1) AS Total_Sales,
       CASE
           WHEN SUM(Sales) > 5000 THEN 'High Spender'
           WHEN SUM(Sales) > 1000 THEN 'Medium Spender'
           ELSE 'Low Spender'
       END AS Spending_Category
FROM orders
GROUP BY Customer_ID
LIMIT 10;


customer_id|total_sales|spending_category|
-----------+-----------+-----------------+
LT-16765   |      329.9|Low Spender      |
SC-20845   |      280.6|Low Spender      |
AC-10615   |     2537.7|Medium Spender   |
BT-11485   |      415.2|Low Spender      |
JP-15520   |     3635.6|Medium Spender   |
CM-11815   |     1673.9|Medium Spender   |
LB-16735   |       50.2|Low Spender      |
JD-16150   |     8828.0|High Spender     |
JG-15805   |     1507.0|Medium Spender   |
TS-21610   |     2820.4|Medium Spender   |
```


### Оконные функции
Найдем топ-3 заказа с наибольшим объемом продаж в каждом регионе
```
SELECT Customer_ID,
       SUM(Sales) AS Total_Sales,
       RANK() OVER (ORDER BY SUM(Sales) DESC) AS Sales_Rank
FROM orders
GROUP BY Customer_ID;


WITH RankedOrders AS (
    SELECT Order_ID,
           Region,
           Sales,
           RANK() OVER (PARTITION BY Region ORDER BY Sales DESC) AS Sales_Rank
    FROM orders
    WHERE Sales IS NOT NULL
      AND Sales > 0
)
SELECT Region,
	   Order_ID,
       ROUND(Sales, 1) AS sales,
       Sales_Rank
FROM RankedOrders
WHERE Sales_Rank <= 3;


region |order_id      |sales  |sales_rank|
-------+--------------+-------+----------+
Central|CA-2018-118689|17500.0|         1|
Central|CA-2018-117121| 9892.7|         2|
Central|CA-2016-116904| 9450.0|         3|
East   |CA-2019-127180|11200.0|         1|
East   |CA-2019-166709|10500.0|         2|
East   |US-2018-107440| 9099.9|         3|
South  |CA-2016-145317|22638.5|         1|
South  |CA-2018-158841| 8750.0|         2|
South  |US-2019-168116| 8000.0|         3|
West   |CA-2019-140151|14000.0|         1|
West   |CA-2016-143917| 8187.7|         2|
West   |CA-2019-135909| 5084.0|         3|
```


Найдем заказы, сумма продаж которых входит в 10 % от общего объема продаж для соответствующих категорий товаров
```
WITH SalesWithTile AS (
    SELECT 
        Order_ID,
        Product_ID,
        Category,
        Sales,
        NTILE(10) OVER (PARTITION BY Category ORDER BY Sales DESC) AS Sales_Tile
    FROM orders
)
SELECT *
FROM SalesWithTile
WHERE Sales_Tile = 1
LIMIT 5;


order_id      |product_id     |category |sales    |sales_tile|
--------------+---------------+---------+---------+----------+
CA-2019-118892|FUR-CH-10002024|Furniture|4416.1740|         1|
CA-2017-117086|FUR-BO-10004834|Furniture|4404.9000|         1|
CA-2017-116638|FUR-TA-10000198|Furniture|4297.6440|         1|
US-2017-126977|FUR-BO-10004834|Furniture|4228.7040|         1|
CA-2016-128209|FUR-BO-10002213|Furniture|4007.8400|         1|
```
