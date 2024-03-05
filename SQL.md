### Базовое извлечение данных
Выводим столбцы из таблицы "Orders" для определенного клиента
```
SELECT *
FROM orders
WHERE Customer_ID = 'SO-20335';

row_id|order_id      |order_date|ship_date |ship_mode     |customer_id|customer_name |segment |country      |city           |state     |postal_code|region|product_id     |category       |subcategory|product_name                                                                   |sales   |quantity|discount|profit               |
------+--------------+----------+----------+--------------+-----------+--------------+--------+-------------+---------------+----------+-----------+------+---------------+---------------+-----------+-------------------------------------------------------------------------------+--------+--------+--------+---------------------+
     4|US-2017-108966|2017-10-11|2017-10-18|Standard Class|SO-20335   |Sean O'Donnell|Consumer|United States|Fort Lauderdale|Florida   |      33311|South |FUR-TA-10000577|Furniture      |Tables     |Bretford CR4500 Series Slim Rectangular Table                                  |957.5775|       5|    0.45|-383.0310000000000000|
     5|US-2017-108966|2017-10-11|2017-10-18|Standard Class|SO-20335   |Sean O'Donnell|Consumer|United States|Fort Lauderdale|Florida   |      33311|South |OFF-ST-10000760|Office Supplies|Storage    |Eldon Fold 'N Roll Cart System                                                 | 22.3680|       2|    0.20|   2.5164000000000000|
  3374|CA-2017-161718|2017-12-04|2017-12-10|Standard Class|SO-20335   |Sean O'Donnell|Consumer|United States|Hempstead      |New York  |      11550|East  |FUR-FU-10002445|Furniture      |Furnishings|DAX Two-Tone Rosewood/Black Document Frame, Desktop, 5 x 7                     | 28.4400|       3|    0.00|  11.3760000000000000|
  3375|CA-2017-161718|2017-12-04|2017-12-10|Standard Class|SO-20335   |Sean O'Donnell|Consumer|United States|Hempstead      |New York  |      11550|East  |FUR-CH-10002372|Furniture      |Chairs     |Office Star - Ergonomically Designed Knee Chair                                |364.4100|       5|    0.10|   8.0980000000000200|
  3376|CA-2017-161718|2017-12-04|2017-12-10|Standard Class|SO-20335   |Sean O'Donnell|Consumer|United States|Hempstead      |New York  |      11550|East  |TEC-PH-10000376|Technology     |Phones     |Square Credit Card Reader                                                      | 39.9600|       4|    0.00|  10.3896000000000000|
  3377|CA-2017-161718|2017-12-04|2017-12-10|Standard Class|SO-20335   |Sean O'Donnell|Consumer|United States|Hempstead      |New York  |      11550|East  |FUR-CH-10002965|Furniture      |Chairs     |Global Leather Highback Executive Chair with Pneumatic Height Adjustment, Black|361.7640|       2|    0.10|  68.3332000000000000|
  4623|CA-2019-147228|2019-09-09|2019-09-14|Standard Class|SO-20335   |Sean O'Donnell|Consumer|United States|Columbia       |Tennessee |      38401|South |OFF-SU-10001225|Office Supplies|Supplies   |Staple remover                                                                 |  8.8320|       3|    0.20|  -1.9872000000000000|
  4624|CA-2019-147228|2019-09-09|2019-09-14|Standard Class|SO-20335   |Sean O'Donnell|Consumer|United States|Columbia       |Tennessee |      38401|South |OFF-PA-10000357|Office Supplies|Paper      |Xerox 1888                                                                     |177.5360|       4|    0.20|  62.1376000000000000|
  4625|CA-2019-147228|2019-09-09|2019-09-14|Standard Class|SO-20335   |Sean O'Donnell|Consumer|United States|Columbia       |Tennessee |      38401|South |OFF-ST-10000046|Office Supplies|Storage    |Fellowes Super Stor/Drawer Files                                               |258.4800|       2|    0.20|  -3.2309999999999900|
  4626|CA-2019-147228|2019-09-09|2019-09-14|Standard Class|SO-20335   |Sean O'Donnell|Consumer|United States|Columbia       |Tennessee |      38401|South |FUR-FU-10000023|Furniture      |Furnishings|Eldon Wave Desk Accessories                                                    | 14.1360|       3|    0.20|   4.2408000000000000|
  6980|CA-2019-149076|2019-01-14|2019-01-19|Standard Class|SO-20335   |Sean O'Donnell|Consumer|United States|Los Angeles    |California|      90036|West  |OFF-PA-10000483|Office Supplies|Paper      |Xerox 19                                                                       |154.9000|       5|    0.00|  69.7050000000000000|
  7122|CA-2019-166926|2019-12-01|2019-12-08|Standard Class|SO-20335   |Sean O'Donnell|Consumer|United States|Seattle        |Washington|      98105|West  |OFF-PA-10001593|Office Supplies|Paper      |Xerox 1947                                                                     | 41.8600|       7|    0.00|  18.8370000000000000|
  7123|CA-2019-166926|2019-12-01|2019-12-08|Standard Class|SO-20335   |Sean O'Donnell|Consumer|United States|Seattle        |Washington|      98105|West  |FUR-BO-10002598|Furniture      |Bookcases  |Hon Metal Bookcases, Putty                                                     |141.9600|       2|    0.00|  41.1684000000000000|
  8592|CA-2019-101700|2019-04-23|2019-04-26|First Class   |SO-20335   |Sean O'Donnell|Consumer|United States|Greeley        |Colorado  |      80634|West  |OFF-EN-10003134|Office Supplies|Envelopes  |Staple envelope                                                                | 18.6880|       2|    0.20|   7.0080000000000000|
  8593|CA-2019-101700|2019-04-23|2019-04-26|First Class   |SO-20335   |Sean O'Donnell|Consumer|United States|Greeley        |Colorado  |      80634|West  |FUR-FU-10001025|Furniture      |Furnishings|Eldon Imàge Series Desk Accessories, Clear                                     | 11.6640|       3|    0.20|   3.3534000000000000|
```










