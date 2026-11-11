import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;

/** Console summaries for character-grouped dialogue data. */
public final class Utilities {
    private Utilities() { }

    /**
     * Prints the total number of dialogue entries, distinct chapters, distinct
     * characters, and up to five randomly selected characters with their entries.
     */
    public static void printSummary(Map<String, List<DialogueCsvReader.DialogueLine>> entriesByCharacter) {
        int lineCount = 0;
        Set<String> chapters = new HashSet<>();
        for (List<DialogueCsvReader.DialogueLine> entries : entriesByCharacter.values()) {
            lineCount += entries.size();
            for (DialogueCsvReader.DialogueLine entry : entries) chapters.add(entry.chapter());
        }

        System.out.println("Books: " + countBooks(entriesByCharacter));
        System.out.println("Chapters: " + chapters.size());
        System.out.println("Characters: " + entriesByCharacter.size());
        System.out.println("Lines: " + lineCount);

        List<String> characters = new ArrayList<>(entriesByCharacter.keySet());
        // Partial Fisher-Yates shuffle gives five distinct random characters.
        int sampleSize = Math.min(5, characters.size());
        for (int i = 0; i < sampleSize; i++) {
            int swapIndex = ThreadLocalRandom.current().nextInt(i, characters.size());
            String temp = characters.get(i);
            characters.set(i, characters.get(swapIndex));
            characters.set(swapIndex, temp);

            String character = characters.get(i);
            List<DialogueCsvReader.DialogueLine> entries = entriesByCharacter.get(character);
            DialogueCsvReader.DialogueLine entry = entries.get(
                    ThreadLocalRandom.current().nextInt(entries.size()));
            System.out.printf("%n%s [%s | %s] %s%n",
                    character, entry.book(), entry.chapter(), entry.line());
        }
    }

    private static int countBooks(Map<String, List<DialogueCsvReader.DialogueLine>> entriesByCharacter) {
        Set<String> books = new HashSet<>();
        for (List<DialogueCsvReader.DialogueLine> entries : entriesByCharacter.values()) {
            for (DialogueCsvReader.DialogueLine entry : entries) books.add(entry.book());
        }
        return books.size();
    }
}
