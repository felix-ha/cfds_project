CREATE TABLE data (
            date DATE,
            value REAL,
            infoID INT
        );
        
CREATE TABLE countries (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            code TEXT
        );

CREATE TABLE types (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            name TEXT
        );

CREATE TABLE info (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            typeID TEXT,
            countryID INT,
            sourceID INT
        );


--ToImplement:
CREATE TABLE sources (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            type TEXT,
            web_url TEXT
        );




        
