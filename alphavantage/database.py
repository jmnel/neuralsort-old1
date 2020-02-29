from abc import ABC, abstractmethod
import sqlite3


class DatabaseConnector(ABC):

    @staticmethod
    @abstractmethod
    def connect(*args, **kwargs):
        pass

#    @abstractmethod
#    def get_tables(self):
#        pass


class SQLite3Connector(DatabaseConnector):

    _connection: sqlite3.Connection = None

    def __init__(self, conn):
        self._connection = conn

    @classmethod
    def connect(cls, db_file: str):
        try:
            conn = sqlite3.connect(db_file)

        except:
            print(e)
            return None

        else:
            return cls(conn)

    def get_tables(self):
        query = 'SELECT name from sqlite_master where type= "table"'
        try:
            cursor = self._connection.cursor()
            result = cursor.execute(query)

        except sqlite3.Error as e:
            print(f'sql error: {e}')

        else:
            tables = list()
            for r in result:
                tables.append(*r)

            return tables

    def table_exists(self, name: str):

        try:
            cursor = self._connection.cursor()
            query = """SELECT * FROM sqlite_master
            WHERE type='table' and NAME=?;"""
            result = cursor.execute(query, [name]).fetchall()

        except sqlite3.Error as e:
            print(f'sql error: {e}')

        else:
            return len(result) > 0

    def get_schema(self, table_name):

        if self.table_exists(table_name):
            cursor = self._connection.cursor()
            query = f'PRAGMA table_info("{table_name}");'
            result = cursor.execute(query)
            schema = list()
            for row in result:
                index, name, dtype, constraints, default, extra = (*row,)
                schema.append({'index': index,
                               'name': name,
                               'dtype': dtype,
                               'constraints': constraints,
                               'default': default,
                               'extra': extra})

            return schema

        else:
            raise ValueError(f'Table {table_name} not found')
            return None


foo = SQLite3Connector.connect('test.db')
tables = foo.get_tables()

s = foo.get_schema('symbols_meta')

for a in s:
    print(a)
    print()

# print(foo.table_exists('symbols_meta'))
