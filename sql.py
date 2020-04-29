
SELECT trackid, name, title FROM tracks INNER JOIN albums ON albums.albumid = tracks.albumid;

SELECT FirstName || ' ' || LastName AS FullNam FROM Employee ORDER BY FullName;

SELECT pat.my_date, pat.l_name, meas.res, meas.msg, pat.sur_reply, pat.insur FROM pat INNER JOIN meas USING(dev_id);