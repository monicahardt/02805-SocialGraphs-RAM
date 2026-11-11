import java.io.IOException;
import java.nio.file.Path;
public class Main {

    public static void main(String[] args) throws IOException {
        Path csvPath = args.length > 0 ? Path.of(args[0]) : Path.of("movie_data/hp1.csv");
        String csvName = csvPath.getFileName().toString();
        int extension = csvName.lastIndexOf('.');
        String name = extension < 0 ? csvName : csvName.substring(0, extension);
        boolean bookData = csvPath.toString().contains("book_data");
        String suffix = bookData ? "_book_networks.html" : "_movie_networks.html";
        Path outputPath = args.length > 1 ? Path.of(args[1]) : Path.of("graphs", name + suffix);
        InteractionGraph.writeHtml(DialogueCsvReader.readDialogueRecords(csvPath), outputPath);
        System.out.println("Wrote character network to " + outputPath);
    }
}
