# proeven_verzameling
BIS database query tool for use within QGIS  
Repository: https://github.com/KRS-dev/proeven_verzameling  
Author: Kevin Schuurman  
E-mail: kevinschuurman98@gmail.com  

## Installation 
1.  Have a PostgreSQL/Oracle Bis database copy running on your PC or know how to connect to one remotely. A PostgreSQL dump can be downloaded from within the repository in the directory 'Database_dump'. The dump is encrypted, e-mail me at kevinschuurman98@gmail.com to get the encryption key. You can install PostgreSQL with a suitable IDE to restore the dump and have the database running on your local pc. The standard install from PostgreSQL comes with PgAdmin which is fine. **Note: The geometry data in graf_loc_aanduidingen requires PostgreSQL to have a spatial extension called PostGIS. The extension can be easily installed through the stack builder that comes with the PostgreSQL install.**
2. Install QGIS
3. Install the needed modules in QGIS. Express installation of QGIS (3.0+) should have most of those pre-installed apart from possibly *pandas* and *cx_Oracle*.
* *pandas*
* *numpy*
* *xlwt*
* *psycop2g*
* *cx_Oracle or pyodbc* (Oracle database)
4. Create a QGIS project with a connection to the database in 1.
5. Import the **BIS_GRAF_LOC_AANDUIDINGEN** into your project.
6. Copy *qgis_frontend.py* and *qgis_backend.py* into your personal python scripts folder.
7. Edit the following line in *qgis_frontend.py* so that it refers to the location where the scripts are located.
```python 
os.chdir(os.path.abspath(r'D:\documents\Github\proeven_verzameling'))
``` 
7. Edit the fetch() function in qgis_backend.py to correctly make a connection with your database:

  ```python
  import psycopg2 as psy
  #import cx_Oracle as cora 
  #or
  #import pyodbc
  def fetch (query, data):
      ## Using a PostgreSQL database
      with psy.connect(
          host = "localhost",
          database = "bis",
          user = "postgres",
          password = "admin"
          ) as dbcon:
      ## Using an Oracle database:
          '''
      with cora.connect(
          user = "",
          password = "",
          dsn = ""*
              ) as dbcon:
          # *Can be a: 
          # 1. Oracle Easy Connect string
          # 2. Oracle Net Connect Descriptor string
          # 3. Net Service Name mapping to connect description
          '''
```
8. Restart QGIS to reload all modules. (Updating *qgis_backend.py*)
## Usage
1. Select your points on **BIS_GRAF_LOC_AANDUIDINGEN**
2. Edit all the inputs in *qgis_frontend.py* to your liking!
3. Run *qgis_frontend.py* inside the QGIS Python shell and you are good to go!
