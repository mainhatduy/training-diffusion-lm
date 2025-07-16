# State 1: Chuẩn hóa dữ liệu (Có nên không?) đây là bước chuẩn hóa lại nội dung trong context để có được sự thống nhất về keyword.

# State 2: NER (bước này thay vì token mask thì sẽ lấy luôn input vào để mô hình xào tiếp)
Input:
    ```
    <ner>
    Chạy bộ thường xuyên có thể giúp cải thiện nhịp tim
    </ner>
    ```
Output:
    ```
    <ner>
    <entity>Chạy bộ thường xuyên</entity> có thể giúp cải thiện <entity>nhịp tim</entity>
    </ner>
    ```
# State 3: RC
Input:
    ```
    <rc>
    <entity>Chạy bộ thường xuyên</entity> có thể giúp cải thiện <entity>nhịp tim</entity>
    </rc>
    ```

Output:
    ```
    {("Chạy bộ thường xuyên", "cải thiện", "nhịp tim")}
    ```

Dạy SFT model 3 kỹ năng. Dùng rule base để thay đổi token nhằm thay đổi task cho DLM sử dụng như một pipeline duy nhất.
(có thể cần hoặc không token kết thúc vì đây là diffusion model cần xem xét)

**Nếu chuẩn hóa được thì quá trình RL sẽ dễ hơn** 