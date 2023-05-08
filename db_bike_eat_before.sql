DROP TABLE IF EXISTS magasin CASCADE;
DROP TABLE IF EXISTS users CASCADE;
DROP TABLE IF EXISTS command CASCADE;

CREATE TABLE magasin(
    id serial,
    nom varchar(255) NOT NULL,
    date_creation date,
    PRIMARY KEY(id)
);

CREATE TABLE users(
    id serial,  
    nom varchar(255),
    prenom varchar(255),
    date_creation date,
    PRIMARY KEY(id)
);

CREATE TABLE command(
    id serial,
    id_mag integer,
    id_user integer, 
    cost int NOT NULL,
    date_creation date,
    FOREIGN KEY (id_mag) REFERENCES magasin(id) ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (id_user) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
    PRIMARY KEY(id)
);


INSERT INTO magasin VALUES (DEFAULT, 'KFC', now());
INSERT INTO magasin VALUES (DEFAULT, 'Chez_Tomy', '2023-02-22');
INSERT INTO magasin VALUES (DEFAULT, 'Sushi_family', '2021-12-02');
INSERT INTO magasin VALUES (DEFAULT, 'Del_Arta', '2022-06-14');

INSERT INTO users VALUES (DEFAULT, 'Joel', now());
INSERT INTO users VALUES (DEFAULT, 'Ellie', '2022-07-23');
INSERT INTO users VALUES (DEFAULT, 'Robert', '2013-08-02');
INSERT INTO users VALUES (DEFAULT, 'Dartagnan', '1611-00-00');

INSERT INTO command VALUES(DEFAULT, 1, 1, 34, now());
INSERT INTO command VALUES(DEFAULT, 1, 1, 34, now());
INSERT INTO command VALUES(DEFAULT, 1, 1, 34, now());
INSERT INTO command VALUES(DEFAULT, 1, 1, 34, now());
INSERT INTO command VALUES(DEFAULT, 1, 1, 34, now());
INSERT INTO command VALUES(DEFAULT, 1, 1, 34, now());
INSERT INTO command VALUES(DEFAULT, 1, 1, 34, now());
INSERT INTO command VALUES(DEFAULT, 1, 1, 34, now());
INSERT INTO command VALUES(DEFAULT, 1, 1, 34, now());
INSERT INTO command VALUES(DEFAULT, 1, 1, 34, now());
