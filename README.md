# eco_image
tools for organization and automatic manipulation of ecological image data sets

```
Timothy James Becker, Derin Gezgin, Jun Yi He Wu, Mary Becker. A framework for river connectivity classification using temporal image
processing and attention based neural networks. arxiv.org/abs/2502.00474. Retrieved from http://arxiv.org/abs/2502.00474.
```

### (1) mysql_connector.MQSL class API
mysql_connector.MSQL class has a simplified interface that utilizes ? chars for injection safe insertion and value setting in queries.

Build a list QS pf dict python objects that have the 'sql' key for mysql statements and an optional 'v' key.  Internally the MSQL class with switch the placeholder ? to the needed %s and convert the list of values into a tuple.  Additional simplified fetures are automatic return in dict format if a server response is obtained and a single commit over the entire QS list for efficient execution of queries. An example of a bulk insert is shown in the following example via the db_test.py file which will perform all CRUD operations in one commit.
```python
#manual connector testing v is an injection safe value list, ? are the positional holders for values
QS = [{'sql':'select * from %s.%s;'%(db,tbl)},
      {'sql':'drop table if exists %s.%s;'%(db,tbl)},
      {'sql':'create table %s.%s(pk int primary key, des text, lat float(10), lon float(10));'%(db,tbl)},
      {'sql':'insert into %s.%s value (?,?,?,?),(?,?,?,?);'%(db,tbl),
       'v':[random.randint(0,10),'Hello',get_pos(),get_pos(),random.randint(10,20),'Goodbye',get_pos(),get_pos()]},
      {'sql':'select * from %s.%s'%(db,tbl)}]

with mysql.MYSQL(host=host,port=port,db=db,uid=uid,pwd=pwd) as dbo:
    dbo.set_SQL_V(QS)
    res = dbo.run_SQL_V()
    print(res)

```
### (2) exif metadata retrieval using the pypy exifread
```python
import exifread
with open(path,'rb') as f: tags = exifread.process_file(f)
```
