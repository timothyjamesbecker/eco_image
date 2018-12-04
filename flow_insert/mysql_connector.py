#MYSQL v 0.1.1, 11/14/2018, Timothy Becker
# MYSQL Connection Factory, wraps and simplifies common workflows

import sys
import getpass
import mysql.connector as msc  # pyodbc not easy to configure on mac, pypyodbc not encoding/decoding

class MYSQL:
    def __init__(self,host,db,port=3306,uid=False,pwd=False): # constructor
        # The MSSQL variables for injection safe connection strings
        self.host   = host  # MYSQL server hostname to connect to
        self.port   = port
        self.db     = db  # db name
        self.uid    = uid
        self.pwd    = pwd
        self.errors = ''
        self.data_errors = ''
        self.SQL    = []
        self.V      = []
        self.start()

    def __enter__(self):
        return self

    # type is that the DB error is generating stange files
    def __exit__(self, type, value, traceback):
        try:
            self.conn.close()
        except RuntimeError:
            print('__exit__():ER1.ODBC')
            self.errors += '__exit__():ER1.ODBC' + '\n'
        except Exception as err:
            print('__exit__():ER2.Unknown_Error: {}'.format(err))
            self.errors += '__exit__():ER2. Unknown Error: {}'.format(err)+'\n'
            pass

    def start(self):
        self.conn = None
        if (not self.uid) and (not self.pwd):
            print('uid: '),
            self.uid = sys.stdin.readline().replace('\n','')
            self.pwd = getpass.getpass(prompt='pwd: ',stream=None).replace('\n','')  # was stream=sys.sdin
        try:  # connection start
            self.conn = msc.connect(host=self.host,database=self.db,port=self.port,user=self.uid,password=self.pwd)
        except RuntimeError:
            print('start():ER3.ODBC')
            self.errors += 'start():ER3.ODBC' + '\n'
        except msc.errors.ProgrammingError:
            print('start():ER4.Connection')
            self.errors += 'start():ER4.Connection' + '\n'
        except Exception as err:
            print('start():ER5.Unknown_Error: {}'.format(err))
            self.errors += 'start():ER5.Unknown_Error: {}'.format(err)+'\n'
            pass

    #takes a list of queries QS: [{'sql':'select * from %s.%s where %s = ?'%('test','test','pk'),'v':[13]]
    #and converts the sql to a mysql.connector 8.13+ version -> ? =? %s and attaches to SQL and V
    #SQL = ['select * from test.test where pk = %s'], V = [(13,)]
    def set_SQL_V(self,QS):
        for q in QS:
            if q.has_key('sql'):
                self.SQL += [q['sql'].replace('?','%s')]
                if q.has_key('v'): self.V += [tuple(q['v'])]
                else:              self.V += [()]
        if len(self.SQL) != len(self.V):
            print('set_SQL_V():ER5.A.Unkown_Internal_Error')
            self.SQL,self.V = [],[]

    def run_SQL_V(self):
        res=[]
        try:
            for i in range(len(self.SQL)):
                cursor = self.conn.cursor(dictionary=True)
                cursor.execute(self.SQL[i],self.V[i])
                if cursor.with_rows: res += [cursor.fetchall()]
                cursor.close()
            self.conn.commit()
        except msc.errors.ProgrammingError as err1:
            print('run_SQL_V():ER6.SQL_Malformed Error: {}'.format(err1))
            self.data_errors += 'ER6.SQL_Malformed Error: {}'.format(err1)+'\n'
        except msc.errors.DataError as err2:
            print('run_SQL_V():ER7.Data_Not_Matching_Template: {}'.format(err2))
            self.data_errors += 'ER7.Data_Not_Matching_Template: {}'.format(err2)+'\n'
        except msc.errors.IntegrityError as err3:
            print('run_SQL_V():ER8.SQL_Constraint_Violation: {}'.format(err3))
            self.data_errors += 'ER8.SQL_Constraint_Violation: {}'.format(err3)+'\n'
        except UnicodeDecodeError as err4:
            print('run_SQL_V():ER9.Unicode_Decoding_Error: {}'.format(err4))
            self.errors+='query():ER9.unicode decoding issues\n'
            self.data_errors += 'ER9.Unicode_Decoding_Error: {}'.format(err4)+'\n'
        except Exception as err5:
            print('run_SQL_V():ER10.Unknown_Error: {}'.format(err5))
            self.data_errors += 'ER10.Unknown_Error: {}'.format(err5)+'\n'
            pass
        return res

    def query(self,sql,v):
        res = {}
        try:  # execute one sql and v
            cursor = self.conn.cursor(dictionary=True)
            cursor.execute(sql,v)
            if cursor.with_rows: res = cursor.fetchall()
            cursor.close()
            self.conn.commit()
        except msc.errors.ProgrammingError as err1:
            print('query():ER6.SQL_Malformed Error: {}'.format(err1))
            self.data_errors += 'ER6.SQL_Malformed Error: {}'.format(err1)[0:800]+'\n'
        except msc.errors.DataError as err2:
            print('query():ER7.Data_Not_Matching_Template: {}'.format(err2))
            self.data_errors += 'ER7.Data_Not_Matching_Template: {}'.format(err2)+'\n'
        except msc.errors.IntegrityError as err3:
            print('query():ER8.SQL_Constraint_Violation: {}'.format(err3))
            self.data_errors += 'ER8.SQL_Constraint_Violation: {}'.format(err3)+'\n'
        except UnicodeDecodeError as err4:
            print('query():ER9.Unicode_Decoding_Error: {}'.format(err4))
            self.errors+='query():ER9.unicode decoding issues\n'
            self.data_errors += 'ER9.Unicode_Decoding_Error: {}'.format(err4)+'\n'
        except Exception as err5:
            print('query():ER10.Unknown_Error: {}'.format(err5))
            self.data_errors += 'ER10.Unknown_Error: {}'.format(err5)+'\n'
            pass
        return res
