import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/** Reads a dialogue CSV and groups each dialogue line by its character. */
public final class DialogueCsvReader {
    private DialogueCsvReader() { }

    /** One dialogue entry together with its source book/movie and chapter. */
    public record DialogueLine(String book, String chapter, String line) { }
    /** One CSV row in its original order, including the speaker. */
    public record DialogueRecord(String book, String chapter, String character, String line, String confidence) { }

    /** Reads all valid dialogue rows without changing their original order. */
    public static List<DialogueRecord> readDialogueRecords(Path csvPath) throws IOException {
        List<DialogueRecord> dialogue = new ArrayList<>();
        try (Reader reader = Files.newBufferedReader(csvPath, StandardCharsets.UTF_8)) {
            CsvRecords records = new CsvRecords(reader);
            List<String> header = records.nextRecord();
            if (header == null) throw new IOException("CSV file is empty: " + csvPath);

            int bookColumn = findColumn(header, "movie");
            if (bookColumn < 0) bookColumn = findColumn(header, "book");
            int chapterColumn = findColumn(header, "chapter");
            int characterColumn = findColumn(header, "character");
            int dialogColumn = findColumn(header, "dialog");
            int confidenceColumn = findColumn(header, "confidence");
            if (bookColumn < 0 || chapterColumn < 0 || characterColumn < 0 || dialogColumn < 0) {
                throw new IOException("CSV must contain book/movie, chapter, character, and dialog columns: " + csvPath);
            }

            List<String> row;
            int lastColumn = Math.max(Math.max(bookColumn, chapterColumn), Math.max(characterColumn, dialogColumn));
            while ((row = records.nextRecord()) != null) {
                if (row.size() <= lastColumn) continue;
                String character = row.get(characterColumn).trim();
                if (!character.isEmpty()) {
                    dialogue.add(new DialogueRecord(row.get(bookColumn), row.get(chapterColumn), character,
                            row.get(dialogColumn), confidenceColumn >= 0 && row.size() > confidenceColumn
                                    ? row.get(confidenceColumn) : "high"));
                }
            }
        }
        return dialogue;
    }

    /** Reads the CSV while retaining book and chapter information for each line. */
    public static Map<String, List<DialogueLine>> readEntriesByCharacter(Path csvPath) throws IOException {
        Map<String, List<DialogueLine>> entriesByCharacter = new LinkedHashMap<>();
        try (Reader reader = Files.newBufferedReader(csvPath, StandardCharsets.UTF_8)) {
            CsvRecords records = new CsvRecords(reader);
            List<String> header = records.nextRecord();
            if (header == null) throw new IOException("CSV file is empty: " + csvPath);

            int bookColumn = findColumn(header, "movie");
            if (bookColumn < 0) bookColumn = findColumn(header, "book");
            int chapterColumn = findColumn(header, "chapter");
            int characterColumn = findColumn(header, "character");
            int dialogColumn = findColumn(header, "dialog");
            if (bookColumn < 0 || chapterColumn < 0 || characterColumn < 0 || dialogColumn < 0) {
                throw new IOException("CSV must contain book/movie, chapter, character, and dialog columns: " + csvPath);
            }

            List<String> row;
            while ((row = records.nextRecord()) != null) {
                int lastColumn = Math.max(Math.max(bookColumn, chapterColumn), Math.max(characterColumn, dialogColumn));
                if (row.size() <= lastColumn) continue;
                String character = row.get(characterColumn).trim();
                if (!character.isEmpty()) {
                    DialogueLine entry = new DialogueLine(row.get(bookColumn), row.get(chapterColumn), row.get(dialogColumn));
                    entriesByCharacter.computeIfAbsent(character, ignored -> new ArrayList<>()).add(entry);
                }
            }
        }
        return entriesByCharacter;
    }

    /**
     * Reads a CSV with {@code character} and {@code dialog} columns and returns
     * each character mapped to their dialogue lines, in file order.
     *
     * @param csvPath path to the CSV file (UTF-8)
     * @return an insertion-ordered map from character names to their lines
     * @throws IOException if the file cannot be read or the required columns are missing
     */
    public static Map<String, List<String>> readLinesByCharacter(Path csvPath) throws IOException {
        Map<String, List<String>> linesByCharacter = new LinkedHashMap<>();

        try (Reader reader = Files.newBufferedReader(csvPath, StandardCharsets.UTF_8)) {
            CsvRecords records = new CsvRecords(reader);
            List<String> header = records.nextRecord();
            if (header == null) {
                throw new IOException("CSV file is empty: " + csvPath);
            }

            int characterColumn = findColumn(header, "character");
            int dialogColumn = findColumn(header, "dialog");
            if (characterColumn < 0 || dialogColumn < 0) {
                throw new IOException("CSV must contain 'character' and 'dialog' columns: " + csvPath);
            }

            List<String> row;
            while ((row = records.nextRecord()) != null) {
                if (row.size() <= Math.max(characterColumn, dialogColumn)) {
                    continue;
                }
                String character = row.get(characterColumn).trim();
                String dialogue = row.get(dialogColumn);
                if (!character.isEmpty()) {
                    linesByCharacter.computeIfAbsent(character, ignored -> new ArrayList<>()).add(dialogue);
                }
            }
        }
        return linesByCharacter;
    }

    private static int findColumn(List<String> header, String wanted) {
        for (int i = 0; i < header.size(); i++) {
            if (header.get(i).trim().toLowerCase(Locale.ROOT).equals(wanted)) {
                return i;
            }
        }
        return -1;
    }

    /** Small RFC 4180-style reader supporting quoted commas, escaped quotes, and newlines. */
    private static final class CsvRecords {
        private final Reader reader;
        private int pending = -2;

        CsvRecords(Reader reader) {
            this.reader = reader;
        }

        List<String> nextRecord() throws IOException {
            List<String> fields = new ArrayList<>();
            StringBuilder field = new StringBuilder();
            boolean inQuotes = false;
            boolean sawAnyInput = false;

            while (true) {
                int value = read();
                if (value == -1) {
                    if (!sawAnyInput && fields.isEmpty() && field.length() == 0) return null;
                    fields.add(field.toString());
                    return fields;
                }
                sawAnyInput = true;
                char c = (char) value;

                if (inQuotes) {
                    if (c == '"') {
                        int next = read();
                        if (next == '"') {
                            field.append('"');
                        } else {
                            inQuotes = false;
                            pending = next;
                        }
                    } else {
                        field.append(c);
                    }
                } else if (c == '"' && field.length() == 0) {
                    inQuotes = true;
                } else if (c == ',') {
                    fields.add(field.toString());
                    field.setLength(0);
                } else if (c == '\n' || c == '\r') {
                    if (c == '\r') {
                        int next = read();
                        if (next != '\n') pending = next;
                    }
                    fields.add(field.toString());
                    return fields;
                } else {
                    field.append(c);
                }
            }
        }

        private int read() throws IOException {
            if (pending != -2) {
                int value = pending;
                pending = -2;
                return value;
            }
            return reader.read();
        }
    }
}
