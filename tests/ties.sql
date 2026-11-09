CREATE TABLE ties (
    id BIGINT PRIMARY KEY,
    v VECTOR(1),
    tag VARCHAR(8)
);

INSERT INTO ties VALUES
    (3, '[1]', 'a'),
    (1, '[1]', 'a'),
    (2, '[1]', 'a'),
    (4, '[2]', 'b'),
    (5, '[1]', 'b');

SELECT id, L2_DISTANCE(v, '[0]') AS d FROM ties ORDER BY d LIMIT 3;
SELECT id FROM ties WHERE tag = 'b' ORDER BY id LIMIT 1;
SELECT id, L2_DISTANCE(v, '[0]') AS d FROM ties WHERE id >= 3 ORDER BY d LIMIT 2;
SELECT id, L2_DISTANCE(v, '[0]') AS d FROM ties ORDER BY d DESC LIMIT 2;
SELECT id FROM ties ORDER BY id DESC LIMIT 2;

CREATE TABLE neg (id BIGINT PRIMARY KEY);
INSERT INTO neg VALUES (-9223372036854775808);
SELECT id FROM neg;

CREATE TABLE s (id BIGINT PRIMARY KEY, t VARCHAR(16));
INSERT INTO s VALUES (1, 'x;y'), (2, 'it''s');
-- this comment contains ; and should be ignored
SELECT id, t FROM s ORDER BY id;
SELECT * FROM s ORDER BY id;
