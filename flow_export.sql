--
-- File generated with SQLiteStudio v3.2.1 on Sat Sep 14 18:32:02 2019
--
-- Text encoding used: UTF-8
--
PRAGMA foreign_keys = off;
BEGIN TRANSACTION;

-- Table: yours
CREATE TABLE yours (hd TEXT, msg TEXT, dev_id INTEGER (16) UNIQUE NOT NULL, pat_id INTEGER UNIQUE PRIMARY KEY AUTOINCREMENT, bed_id INTEGER (16), sig CHAR (32) UNIQUE, res CHAR (16));
INSERT INTO yours (hd, msg, dev_id, pat_id, bed_id, sig, res) VALUES ('DA', 'Abnormal', 1234, 1, NULL, '132pre1.wav', NULL);
INSERT INTO yours (hd, msg, dev_id, pat_id, bed_id, sig, res) VALUES ('DA', 'Abnormal', 1237, 2, 12, '135pre1.wav', NULL);

COMMIT TRANSACTION;
PRAGMA foreign_keys = on;
