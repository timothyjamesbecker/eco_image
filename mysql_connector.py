#MYSQL v 0.1.1, 11/14/2018, Timothy Becker
# MYSQL Connection Factory, wraps and simplifies common workflows

import sys
import getpass
import mysql.connector as msc  # pyodbc not easy to configure on mac, pypyodbc not encoding/decoding

class MYSQL:
    def __init__(self,host,db,port=3306,uid=False,pwd=False,
                 ssh_tunnel=False,ssh_uid=False,ssh_pwd=False,ssh_port=22,
                 delim='?',charset=['utf8','utf8_general_ci']):
        self.host       = host       # hostname to connect to: either MySQL server host or ssh
        self.db         = db         # MySQL db schema name
        self.port       = port       # MySQL db port
        self.charset    = charset    # MySQL charset encoding for text,varchar
        self.uid        = uid        # MySQL username
        self.pwd        = pwd        # MySQL password
        self.ssh_port   = ssh_port   # optional ssh port
        self.ssh_uid    = ssh_uid    # optional ssh username
        self.ssh_pwd    = ssh_pwd    # optional ssh password
        self.ssh_tunnel = ssh_tunnel # use of optional ssh
        self.tunnel     = None       # session attach point
        self.delim      = delim      # the delimiter char pattern to switch for injection
        self.errors     = ''         # error string for possible logging options
        self.SQL        = []         # sql statement container
        self.V          = []         # value container
        self.start()                 # start a connection

    def __enter__(self):
        return self

    # type is that the DB error is generating stange files
    def __exit__(self, type, value, traceback):
        try:
            print('closing the connection')
            self.conn.close()
            if self.tunnel is not None:
                print('closing ssh tunnel...')
                self.tunnel.stop()
                print('tunnel has been closed...')
        except RuntimeError:
            print('__exit__():ER1.ODBC')
            self.errors += '__exit__():ER1.ODBC' + '\n'
        except Exception as err:
            print('__exit__():ER2.Unknown_Error: {}'.format(err))
            self.errors += '__exit__():ER2. Unknown Error: {}'.format(err)+'\n'
            pass

    def start(self):
        self.conn = None
        if self.ssh_tunnel:
            import sshtunnel
            if (not self.ssh_uid) and (not self.ssh_pwd):
                print('ssh_uid: '),
                self.ssh_uid = sys.stdin.readline().replace('\n','')
                self.ssh_pwd = getpass.getpass(prompt='ssh_pwd: ',stream=None).replace('\n','')
            if (not self.uid) and (not self.pwd):
                print('db_uid: '),
                self.uid=sys.stdin.readline().replace('\n','')
                self.pwd=getpass.getpass(prompt='db_pwd: ',stream=None).replace('\n','')  # was stream=sys.sdin
            print('opening ssh tunnel...')
            self.tunnel = sshtunnel.SSHTunnelForwarder(
                    (self.host,self.ssh_port),
                    ssh_username=self.ssh_uid,
                    ssh_password=self.ssh_pwd,
                    remote_bind_address=('127.0.0.1',self.port),
                    local_bind_address=('0.0.0.0',self.port)
            )
            self.tunnel.start()
            try:
                self.conn = msc.connect(host='127.0.0.1',database=self.db,port=self.port,
                                        user=self.uid,password=self.pwd)
                self.conn.set_charset_collation(self.charset[0],self.charset[1])
                print('tunnel established...')
            except RuntimeError:
                print('start():ER3.ODBC')
                self.errors+='start():ER3.ODBC'+'\n'
            except msc.errors.ProgrammingError:
                print('start():ER4.Connection')
                self.errors+='start():ER4.Connection'+'\n'
            except Exception as err:
                print('start():ER5.Unknown_Error: {}'.format(err))
                self.errors+='start():ER5.Unknown_Error: {}'.format(err)+'\n'
                pass
        else:
            if (not self.uid) and (not self.pwd):
                print('db_uid: '),
                self.uid = sys.stdin.readline().replace('\n','')
                self.pwd = getpass.getpass(prompt='db_pwd: ',stream=None).replace('\n','')  # was stream=sys.sdin
            try:  # connection start
                self.conn = msc.connect(host=self.host,database=self.db,port=self.port,user=self.uid,password=self.pwd)
                self.conn.set_charset_collation(self.charset[0],self.charset[1])
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
                self.SQL += [q['sql'].replace(self.delim,'%s')]
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
        except msc.errors.DataError as err2:
            print('run_SQL_V():ER7.Data_Not_Matching_Template: {}'.format(err2))
        except msc.errors.IntegrityError as err3:
            print('run_SQL_V():ER8.SQL_Constraint_Violation: {}'.format(err3))
        except UnicodeDecodeError as err4:
            print('run_SQL_V():ER9.Unicode_Decoding_Error: {}'.format(err4))
            self.errors+='query():ER9.unicode decoding issues\n'
        except Exception as err5:
            print('run_SQL_V():ER10.Unknown_Error: {}'.format(err5))
            pass
        return res

    def query(self,sql,v):
        res = {}
        try:  # execute one sql and v
            cursor = self.conn.cursor(dictionary=True)
            cursor.execute(sql.replace(self.delim,'%s'),tuple(v))
            if cursor.with_rows: res = cursor.fetchall()
            cursor.close()
            self.conn.commit()
        except msc.errors.ProgrammingError as err1:
            print('run_SQL_V():ER6.SQL_Malformed Error: {}'.format(err1))
        except msc.errors.DataError as err2:
            print('run_SQL_V():ER7.Data_Not_Matching_Template: {}'.format(err2))
        except msc.errors.IntegrityError as err3:
            print('run_SQL_V():ER8.SQL_Constraint_Violation: {}'.format(err3))
        except UnicodeDecodeError as err4:
            print('run_SQL_V():ER9.Unicode_Decoding_Error: {}'.format(err4))
            self.errors+='query():ER9.unicode decoding issues\n'
        except Exception as err5:
            print('run_SQL_V():ER10.Unknown_Error: {}'.format(err5))
            pass
        return res
